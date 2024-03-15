# [Nash Learning from Human Feedback](http://arxiv.org/abs/2312.00886)


import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, LambdaLR
from tqdm import tqdm


USE_ATARI = 0
USE_NLP = ~USE_ATARI

if USE_NLP:
    # only enables one of the following
    USE_BERT = 1
    USE_CAUSAL_LM = 0
    USE_SEQ2SEQ_LM = 0
    USE_MAMBA = 0

USE_ADAMW_ON_LION = 1
USE_ADAMW = ~USE_ADAMW_ON_LION


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16


torch.autograd.set_detect_anomaly(True)
debugging_is_on = False


def print_tensor_info(tensor_name, tensor):
    # Check if tensor is floating point, and convert if necessary
    tensor_float = tensor.float() if not tensor.is_floating_point() else tensor

    # Gather the information
    info = {
        "shape": tuple(tensor.shape),
        "min/max": (tensor.min().item(), tensor.max().item()),
        "mean": tensor_float.mean().item(),
        "std": tensor_float.std().item()
    }

    # Print the default representation and the extra information
    print(f"{tensor_name} = {tensor}")
    for key, value in info.items():
        print(f"{key}: {value}")


# Adjusting beta affects the balance between exploiting the current policy
# and exploring new policies suggested by the reference model or feedback.
beta = 0.5
tau = 1 - beta

# alpha is the weight on how to linearly combine the preference losses
alpha = 0.5


"""
In reinforcement learning, a current policy and a reference policy represent
different strategies for decision-making.

The current policy (π) is the strategy the agent is currently using to make
decisions. It is typically represented as a probability distribution over
actions given states.

The reference policy (μ), sometimes used as a baseline or target, can be a
previous version of the current policy, a fixed heuristic, or an expert policy.

both policies are instances of the same neural network class, but they could be
separate models or even different types of models depending on the NLHF
application. The current policy is what we actively train, and the reference
policy provides a stable comparison point which could be static or
periodically updated.
"""

if USE_ATARI:
    class PolicyNetwork(nn.Module):
        def __init__(self, input_size, output_size):
            super(PolicyNetwork, self).__init__()
            # Define network layers
            self.layers = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Linear(256, output_size),
                nn.Softmax(dim=-1)
            )

        def forward(self, state):
            y = self.layers(state)
            print(f"y has shape of {y.shape}")
            return y

elif USE_NLP:
    if USE_BERT:
        from transformers import AutoModel
    elif USE_CAUSAL_LM or USE_MAMBA:
        from transformers import AutoModelForCausalLM
    elif USE_SEQ2SEQ_LM:
        from transformers import AutoModelForSeq2SeqLM

    # Create model instance
    if USE_BERT:
        bert_model = AutoModel.from_pretrained("prajjwal1/bert-tiny").to(device)
    elif USE_CAUSAL_LM or USE_MAMBA:
        # bert_model = AutoModelForCausalLM.from_pretrained("ContextualAI/archangel_sft-dpo_llama7b")
        bert_model = AutoModelForCausalLM.from_pretrained("AdamG012/chat-opt-350m-reward-deepspeed")
        # bert_model = AutoModelForCausalLM.from_pretrained("Q-bert/Mamba-370M", trust_remote_code=True)
    elif USE_SEQ2SEQ_LM:
        bert_model = AutoModelForSeq2SeqLM.from_pretrained("stanfordnlp/SteamSHP-flan-t5-large")

    # for attr, value in bert_model.config.__dict__.items():
    #     print(f"{attr}: {value}")

    print(f"bert_model.config.hidden_size = {bert_model.config.hidden_size}")

    class PolicyNetwork(nn.Module):
        def __init__(self, state_dim):
            super(PolicyNetwork, self).__init__()
            self.bert = bert_model
            self.final_layer = nn.Linear(self.bert.config.hidden_size, 1)
            self.dropout = nn.Dropout(0.2)  # Add dropout
            self.relu = nn.ReLU()

        def forward(self, input_ids, attention_mask=None):
            if USE_BERT:
                # Process text
                text_embeddings = self.bert(
                                    input_ids,
                                    attention_mask=attention_mask
                                  ).last_hidden_state[:, 0, :]

            elif USE_CAUSAL_LM or USE_MAMBA:
                if USE_MAMBA:
                    # Process text
                    text_embeddings = self.bert(
                                        input_ids,
                                      )

                    # print(type(text_embeddings[0]))   # Should print <class 'torch.Tensor'>
                    # print(text_embeddings[0].shape)   # Let's see its dimensions

                    text_embeddings = text_embeddings[0][:, -1, :]  # Last token logits

                else:
                    # Process text
                    text_embeddings = self.bert(
                                        input_ids,
                                        attention_mask=attention_mask
                                      )

                    text_embeddings = text_embeddings.logits[:, -1, :]  # Last token logits
                    # text_embeddings = text_embeddings.logits.mean(dim=1)  # Mean pooling

                text_embeddings = self.dropout(text_embeddings)  # Apply dropout

                intermediate_layer = nn.Linear(text_embeddings.shape[-1],
                                               self.bert.config.hidden_size)

                text_embeddings = intermediate_layer(text_embeddings)

            # Combine and score
            score = self.final_layer(text_embeddings)

            # for numerical stability
            # score = self.relu(score) + 1e-6
            score = torch.sigmoid(score)

            return score


"""
Implementing a preference model function in Python for a reinforcement
learning context involves comparing the expected rewards or values of
different actions and selecting the one that aligns best with human
preferences.

Preference_model uses the probabilities assigned by the current policy
model to each action as a measure of the model's confidence. It then
combines this with a human preference score, which could be obtained
from pre-recorded data or real-time feedback, to produce a final
preference score for the action. This is a simplified version, and
in practice, the human preference component might involve more complex
methods like comparing against a database of preferred actions, or
using a learned model to predict human preferences.

Please note that the function action_to_index(action) would need to be
defined according to how actions are represented in Atari environment,
and human_preferences would be a data structure we'd need to define
based on how we're collecting and storing human feedback.

See Section 7.2 inside the NLHF paper for an overview

See also expression (1.1), section 4 and Theorem 1 of
[Transforming and Combining Rewards for Aligning Large Language Models]
(https://arxiv.org/abs/2402.00742)
"""

def alternative_policy(current_policy_probs, reference_policy_probs):
    # Calculate the alternative policy 𝜋′ using equation (11)
    log_alternative_policy_prob = (1 - beta) * torch.log(current_policy_probs) + \
                                beta * torch.log(reference_policy_probs)

    alternative_policy_prob = torch.exp(log_alternative_policy_prob)

    """
    See section 7 of the paper

    In equation (11), the normalization constant c is indeed important.
    It ensures that after updating the policy using the Nash-MD-PG algorithm,
    the updated policy π is still a valid probability distribution.

    The constant c is determined after the update so that the
    sum of probabilities across all possible actions y equals 1.
    """
    # Normalize the alternative policy 𝜋′ using equation (11)
    alternative_policy_prob = alternative_policy_prob / alternative_policy_prob.sum(dim=-1, keepdim=True)

    """
    The following needs some pondering and may need to be revisited.

    Notice that the one-step-at-a-time regularized policy 𝜋𝜏 𝜃(𝑦|𝑥)
    is different from the original regularized policy 𝜋𝜏 𝜃(𝑦|𝑥)
    because the sequence of normalization constants 𝐶(𝑥, 𝑦0:𝑛−1)
    depend on the specific sample path 𝑦0:𝑛−1 and does not necessarily
    correspond to the full normalization constant 𝑐 defined in Equation (5).

    Credit: Quoted directly from section 7.1 of NLHF paper
    """

    assert alternative_policy_prob >= 0
    assert alternative_policy_prob <= 1

    return alternative_policy_prob


def preference_model(state, state_action_a, state_action_b, mask_a, mask_b,
                     model, reference_model, human_preferences):
    """
    A model that scores actions based on a combination of model predictions
    and human preferences.

    Args:
        state (tensor): The current state from the environment
        state_action_a (tensor): The state-action pair A.
        state_action_b (tensor): The state-action pair B.
        mask: attention mask due to the use of padding
        model (nn.Module): The current model 𝜋𝜃(𝑦|𝑥)).
        reference_model (nn.Module): The reference model 𝜇(𝑦|𝑥).
        human_preferences (tensor): The human preferences.

    return: Preference losses for actions from current policy and alternative policy
    """

    # Use float32 since the models internal compute ops are floating-point
    state = state.float()

    if USE_MAMBA:
        # Get the current policy's probability distribution for actions
        # Sample actions from the current policy 𝜋𝜃
        current_policy_probs_a = model(state_action_a)
        current_policy_probs_b = model(state_action_b)

        # Get the reference policy's probability distribution for actions
        reference_policy_probs_a = reference_model(state_action_a)
        reference_policy_probs_b = reference_model(state_action_b)

    else:
        # Get the current policy's probability distribution for actions
        current_policy_probs_a = model(state_action_a, mask_a)
        current_policy_probs_b = model(state_action_b, mask_b)

        # Get the reference policy's probability distribution for actions
        reference_policy_probs_a = reference_model(state_action_a, mask_a)
        reference_policy_probs_b = reference_model(state_action_b, mask_b)


    # Calculate the log probability of the current policy 𝜋𝜃
    log_current_prob_a = torch.log(current_policy_probs_a)
    log_current_prob_b = torch.log(current_policy_probs_b)

    # Calculate the log probability of the reference policy 𝜇
    log_reference_prob_a = torch.log(reference_policy_probs_a)
    log_reference_prob_b = torch.log(reference_policy_probs_b)

    # Calculate the cross-entropy loss for the current policy
    current_loss_a = -torch.log(current_policy_probs_a)
    current_loss_b = -torch.log(current_policy_probs_b)

    # See equation (9)
    # KL divergence regularization term helps to prevent the current policy
    # from diverging too much from the reference policy during the optimization
    # kl_regularization is proportional to torch.log(current_policy_prob / reference_policy_prob)
    current_kl_divergence_a = log_current_prob_a - log_reference_prob_a
    current_kl_divergence_b = log_current_prob_b - log_reference_prob_b

    """
    The generation of a token at a particular position in the sequence,
    says [A, B, C, D] depends on the previous tokens and the input context.
    The future tokens in the sequence do not directly influence the
    generation of the current token. However, the KL estimator terms for
    the future tokens still provide relevant information for estimating the
    gradient of the current token.

    The reason is that the KL estimator terms capture the discrepancy
    between the learned policy 𝜋𝜃 and the reference policy 𝜇 for the entire
    sequence. By considering the KL estimator terms for the current token
    and the subsequent tokens, we take into account how the current token's
    generation affects the overall sequence and its alignment with the
    reference policy.

    In the example of estimating the gradient for token B (index 𝑛 = 1),
    multiplying the gradient term ∇𝜃 log 𝜋𝜃(B|A) by the KL estimator terms
    for tokens B, C, and D (indices 𝑚 ≥ 1) helps to capture the impact of
    generating token B on the subsequent tokens and the overall sequence.

    While the generation of token B does not directly depend on the future
    tokens C and D, the KL estimator terms for tokens C and D provide info
    about how well the generated sequence aligns with the reference policy.

    By including these terms in the gradient estimate for token B, we
    account for the long-term impact of generating token B on the quality
    of the entire sequence.

    The variance-reduction trick of multiplying the gradient term by the
    KL estimator terms corresponding to indices at least as great as 𝑛
    helps to reduce the variance of the gradient estimate by focusing on
    the relevant information from the current and subsequent tokens,
    while reducing the influence of irrelevant information from the
    previous tokens.

    Credit: Claude-3-Opus-200k AI chatbot
    """

    # Initialize the gradient term for the current policy 𝜋𝜃 using equation (9)
    # current_gradient_term_a = torch.zeros_like(log_current_prob_a)
    # current_gradient_term_b = torch.zeros_like(log_current_prob_b)

    current_kl_estimator_a = None
    current_kl_estimator_b = None
    for n in range(log_current_prob_a.size(1)):
        """
        In practice, when the response 𝑦 comprises a sequence of tokens 𝑦0:𝑁,
        a sample-based estimator to the KL based on the sample response 𝑦 can be used.

        Further, this can be decomposed into a sum across token indicies of per-token
        KL estimators, and the standard policy-gradient variance-reduction trick of
        only multiplying ∇𝜃 log 𝜋𝜃( 𝑦𝑛|𝑥, 𝑦0:𝑛−1) by KL estimator terms corresponding
        to indices at least as great as 𝑛 can be applied.

        Credit: Quoted directly from section 7.2 of NLHF paper
        """
        # Calculate the gradient term using equation (9)
        # Realized that gradient term may not needed after all since both
        # equations (9) and (10) are implicitly realized within autograd engine.
        # current_gradient_term_a[:, n] = torch.autograd.grad(log_current_prob_a[:, n], model.parameters(), retain_graph=True)[0]
        # current_gradient_term_b[:, n] = torch.autograd.grad(log_current_prob_b[:, n], model.parameters(), retain_graph=True)[0]

        # Apply the variance-reduction trick
        current_kl_estimator_a = torch.sum(current_kl_divergence_a[:, n:], dim=1)
        current_kl_estimator_b = torch.sum(current_kl_divergence_b[:, n:], dim=1)
        # current_gradient_term_a[:, n] *= current_kl_estimator_a
        # current_gradient_term_b[:, n] *= current_kl_estimator_b


    # Calculate the preference loss using equation (9)
    '''
    (1 - human_preferences) inverts the preference,
    so when a value of 1 indicates a preference for state-action pair (b),
    a value of 0 would indicate a preference for state-action pair (a).

    baseline = 0.5 is subtracted to center the preference around 0
    Subtract the baseline (average preference score) for variance reduction
    to reduce the variance of the policy gradient estimate, which can help
    stabilize training
    we do not use preference_score.mean(), see equation (9) in NLHF paper

    tau * kl_divergence is the KL divergence regularization term
    for state-action pair, weighted by the coefficient tau.

    So equation (9) effectively maximizes the probability of the
    preferred state-action pair while minimizing the probability
    of the less preferred pair.

    It encourages the policy to stay close to the reference policy
    for the state-action pair.
    '''
    baseline = 0.5

    # Calculate the preference loss using the cross-entropy losses as well as
    # the expectation over states and actions using equation (10)
    # preference_loss = torch.mean(((human_preferences - baseline - tau * current_kl_estimator_a) * current_gradient_term_a +
    #                              ((1 - human_preferences) - baseline - tau * current_kl_estimator_b) * current_gradient_term_b)) / 2)
    current_preference_loss = torch.mean(((human_preferences - baseline - tau * current_kl_estimator_a) * current_loss_a +
                                 ((1 - human_preferences) - baseline - tau * current_kl_estimator_b) * current_loss_b) / 2)


    # Calculate the alternative policy 𝜋′ for state-action pairs (a) and (b) using equation (11)
    alternative_policy_prob_a = alternative_policy(current_policy_probs_a, reference_policy_probs_a)
    alternative_policy_prob_b = alternative_policy(current_policy_probs_b, reference_policy_probs_b)

    # Incorporate the alternative policy 𝜋′ into the expectation defined in equation (10)
    log_alternative_prob_a = torch.log(alternative_policy_prob_a)
    log_alternative_prob_b = torch.log(alternative_policy_prob_b)

    # Calculate the cross-entropy loss for the alternative policy
    alternative_loss_a = -torch.log(alternative_policy_prob_a)
    alternative_loss_b = -torch.log(alternative_policy_prob_b)

    # See equation (9)
    # KL divergence regularization term helps to prevent the alternative policy
    # from diverging too much from the reference policy during the optimization
    # kl_regularization is proportional to torch.log(alternative_policy_prob / reference_policy_prob)
    alternative_kl_divergence_a = log_alternative_prob_a - log_reference_prob_a
    alternative_kl_divergence_b = log_alternative_prob_b - log_reference_prob_b

    # Initialize the gradient term for the alternative policy 𝜋′ using equation (9)
    # alternative_gradient_term_a = torch.zeros_like(log_alternative_prob_a)
    # alternative_gradient_term_b = torch.zeros_like(log_alternative_prob_b)

    alternative_kl_estimator_a = None
    alternative_kl_estimator_b = None
    for n in range(log_alternative_prob_a.size(1)):
        """
        In practice, when the response 𝑦 comprises a sequence of tokens 𝑦0:𝑁,
        a sample-based estimator to the KL based on the sample response 𝑦 can be used.

        Further, this can be decomposed into a sum across token indicies of per-token
        KL estimators, and the standard policy-gradient variance-reduction trick of
        only multiplying ∇𝜃 log 𝜋𝜃( 𝑦𝑛|𝑥, 𝑦0:𝑛−1) by KL estimator terms corresponding
        to indices at least as great as 𝑛 can be applied.

        Credit: Quoted directly from section 7.2 of NLHF paper
        """
        # Calculate the gradient term using equation (9)
        # Realized that gradient term may not needed after all since both
        # equations (9) and (10) are implicitly realized within autograd engine.
        # alternative_gradient_term_a[:, n] = torch.autograd.grad(log_alternative_prob_a[:, n], model.parameters(), retain_graph=True)[0]
        # alternative_gradient_term_b[:, n] = torch.autograd.grad(log_alternative_prob_b[:, n], model.parameters(), retain_graph=True)[0]

        # Apply the variance-reduction trick
        alternative_kl_estimator_a = torch.sum(alternative_kl_divergence_a[:, n:], dim=1)
        alternative_kl_estimator_b = torch.sum(alternative_kl_divergence_b[:, n:], dim=1)
        # alternative_gradient_term_a[:, n] *= alternative_kl_estimator_a
        # alternative_gradient_term_b[:, n] *= alternative_kl_estimator_b


    # Calculate the preference loss using the cross-entropy losses as well as
    # the expectation over states and actions using equation (10)
    # alternative_preference_loss = torch.mean(((human_preferences - baseline - tau * alternative_kl_estimator_a) * alternative_gradient_term_a +
    #                                          ((1 - human_preferences) - baseline - tau * alternative_kl_estimator_b) * alternative_gradient_term_b)) / 2)
    alternative_preference_loss = torch.mean(((human_preferences - baseline - tau * alternative_kl_estimator_a) * alternative_loss_a +
                                             ((1 - human_preferences) - baseline - tau * alternative_kl_estimator_b) * alternative_loss_b) / 2)

    assert current_preference_loss.max() <= 1
    assert alternative_preference_loss.max() <= 1

    return current_preference_loss, alternative_preference_loss


# Dataset selection
if USE_ATARI:
    import agc.dataset as ds
    import agc.util as util

    # DATA_DIR is the directory, which contains the 'trajectories' and
    # 'screens' folders
    DATA_DIR = ""
    dataset = ds.AtariDataset(DATA_DIR)

    # dataset.trajectories returns the dictionary with all the trajs from
    # the Atari dataset
    all_trajectories = dataset.trajectories

    from IPython.display import display, HTML
    from prettytable import PrettyTable

    titles = [' '] + [util.TITLES[g] for g in util.GAMES]

    table = PrettyTable(titles)
    table.align[''] = "l"
    table.align[''] = "l"

    row = ['episodes']
    for g in util.GAMES:
        row.append(dataset.stats[g]['total_replays'])
    table.add_row(row)

    row = ['frames']
    for g in util.GAMES:
        row.append(dataset.stats[g]['total_frames'])
    table.add_row(row)

    row = ['hours of gameplay']
    for g in util.GAMES:
        hrs = float(dataset.stats[g]['total_frames']//60//60/60)
        row.append('%.2f' % (hrs,))
    table.add_row(row)

    row = ['worst score']
    for g in util.GAMES:
        row.append(dataset.stats[g]['min_score'])
    table.add_row(row)

    row = ['best score']
    for g in util.GAMES:
        row.append(dataset.stats[g]['max_score'])
    table.add_row(row)

    row = ['average score']
    for g in util.GAMES:
        row.append("%.0f" % dataset.stats[g]['avg_score'])
    table.add_row(row)

    row = ['score SEM']
    for g in util.GAMES:
        row.append("%.0f" % dataset.stats[g]['sem'])
    table.add_row(row)

    row = ['score stddev']
    for g in util.GAMES:
        row.append("%.0f" % dataset.stats[g]['stddev'])
    table.add_row(row)

    display(HTML(table.get_html_string()))

    # We are using the Atari Pong game environment
    import gym
    # print(f"list of all gym environments = {gym.envs.registry.keys()}")
    env = gym.make('Pong-v4')
    state = env.reset()  # Reset the environment to get the initial state

    # Assuming the first element of the tuple is the screen image we want
    screen = state[0] if isinstance(state, tuple) else state

    # Add batch dimension
    state_tensor = torch.tensor(screen, dtype=torch.float32).unsqueeze(0)
    state_size = 33600


elif USE_NLP:

    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Load all the data
    # dataset = load_dataset("stanfordnlp/shp")

    # Load one of the subreddits
    dataset = load_dataset("stanfordnlp/shp", data_dir="explainlikeimfive")

    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # tokenizer = AutoTokenizer.from_pretrained("ContextualAI/archangel_sft-dpo_llama7b")
    # tokenizer = AutoTokenizer.from_pretrained("AdamG012/chat-opt-350m-reward-deepspeed")
    tokenizer = AutoTokenizer.from_pretrained("stanfordnlp/SteamSHP-flan-t5-large")

    def preprocess_shp_entry(entry):

        state = tokenizer(entry['history'],
                          padding='max_length',
                          truncation=True,
                          max_length=512,
                          return_tensors="pt")

        state_action_a = tokenizer(entry['history'] + entry['human_ref_A'],
                                   padding='max_length',
                                   truncation=True,
                                   max_length=512,
                                   return_tensors="pt")

        state_action_b = tokenizer(entry['history'] + entry['human_ref_B'],
                                   padding='max_length',
                                   truncation=True,
                                   max_length=512,
                                   return_tensors="pt")

        # Indicates preference between action_a and action_b
        preference = entry['labels']
        return state, state_action_a, state_action_b, preference

    encoded_inputs_file = 'encoded_inputs_transformer.pt'

    if os.path.exists(encoded_inputs_file):
        print("Loading pre-tokenized data...")
        encoded_inputs = torch.load(encoded_inputs_file)
    else:
        # Process data
        print("Tokenizing data now ...")
        encoded_inputs = [preprocess_shp_entry(entry)
                          for entry in dataset['train']]
        torch.save(encoded_inputs, encoded_inputs_file)
        print("Finished tokenizing data !!!")

    for item in encoded_inputs:
        state, state_action_a, state_action_b, preference = item

    state_tensor = state['input_ids']
    state_size = state_tensor.size()[-1]


print(f"state has type {type(state)} and length of {len(state)}")
print(f"state_tensor has shape of {state_tensor.size()}")

if USE_ATARI:
    action_size = 64

    # Initialize current policy π to obtain an action from the current policy
    current_policy = PolicyNetwork(input_size=state_size,
                                   output_size=action_size)

    # Initialize reference policy μ (could be a previous checkpoint
    # of the current policy)
    reference_policy = PolicyNetwork(input_size=state_size,
                                     output_size=action_size)
    # reference_policy.load_state_dict(torch.load('path_to_checkpoint'))

elif USE_NLP:
    # Initialize current policy π to obtain an action from the current policy
    current_policy = PolicyNetwork(state_dim=state_size)

    # Initialize reference policy μ (could be a previous checkpoint
    # of the current policy)
    reference_policy = PolicyNetwork(state_dim=state_size)
    # reference_policy.load_state_dict(torch.load('path_to_checkpoint'))

# Set reference policy to evaluation mode if it's not being trained
reference_policy.eval()

# Extracting token IDs for state, action_a and action_b
state_ids = state['input_ids']
state_action_a_ids = state_action_a['input_ids']
state_action_b_ids = state_action_b['input_ids']
state_action_a_mask = state_action_a['attention_mask']
state_action_b_mask = state_action_b['attention_mask']

print("state_ids shape:", state_ids.shape)
print("state_action_a_ids shape:", state_action_a_ids.shape)
print("state_action_b_ids shape:", state_action_b_ids.shape)


class SHPDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['preference'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.data.items()}
        return item


# Combine into a single dictionary
data = {
    'state': state_ids,
    'state_action_a': state_action_a_ids,
    'state_action_b': state_action_b_ids,
    'mask_a': state_action_a_mask,
    'mask_b': state_action_b_mask,
    'preference': torch.tensor(preference).unsqueeze(0)
}


# Split the data into train and validation sets
total_size = len(dataset['train']['labels'])
train_size = int(total_size * 0.8)
print(f"total_size = {total_size}")

train_data = {key: val[:train_size] for key, val in data.items()}
val_data = {key: val[train_size:] for key, val in data.items()}

train_dataset = SHPDataset(train_data)
val_dataset = SHPDataset(val_data)


# Create a DataLoader for batch processing
# Now we can use data_loader in the training loop
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Define the loss function and optimizer
if USE_ADAMW_ON_LION:

    # Credit : https://github.com/google/automl/blob/master/lion/lion_pytorch.py
    # Copyright 2023 Google Research. All Rights Reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ==============================================================================
    """PyTorch implementation of the Lion optimizer."""
    # import torch
    from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable

    class Lion(Optimizer):
        r"""Implements Lion algorithm."""

        def __init__(self, params, lr=1e-4, betas=(0.9, 0.99),
                     weight_decay=0.0, differentiable=True):
            """Initialize the hyperparameters.

            Args:
              params (iterable): iterable of parameters to optimize or
                                 dicts defining parameter groups
              lr (float, optional): learning rate (default: 1e-4)
              betas (Tuple[float, float], optional): coefficients used
                                for computing running averages of gradient
                                and its square (default: (0.9, 0.99))
              weight_decay (float, optional): weight decay coefficient
                                                (default: 0)
            """

            defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
            super().__init__(params, defaults)
            self.defaults['differentiable'] = differentiable  # Initialize

            if not 0.0 <= lr:
                raise ValueError('Invalid learning rate: {}'.format(lr))
            if not 0.0 <= betas[0] < 1.0:
                raise ValueError('Invalid beta parameter at index 0: {}'
                                 .format(betas[0]))
            if not 0.0 <= betas[1] < 1.0:
                raise ValueError('Invalid beta parameter at index 1: {}'
                                 .format(betas[1]))

        @_use_grad_for_differentiable
        def step(self, closure=None):
            """Performs a single optimization step.

            Args:
                closure (callable, optional): A closure that reevaluates
                                            the model and returns the loss.

            Returns:
              the loss.
            """
            loss = None
            updates = []
            if closure is not None:
                loss = closure()

            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        # print("p.grad is None")
                        continue

                    # Perform stepweight decay
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                    grad = p.grad
                    state = self.state[p]
                    # State initialization
                    if len(state) == 0:
                        # print("len(state) == 0:")
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p)

                    exp_avg = state['exp_avg']
                    beta1, beta2 = group['betas']

                    # print(f"exp_avg = {exp_avg}, grad = {grad}")

                    # Weight update
                    update = exp_avg * beta1 + grad * (1 - beta1)

                    # Store the update
                    # updates.append(-torch.sign(update))
                    updates.append(-update.sign_())

                    p.add(update.sign_(), alpha=-group['lr'])

                    # Decay the momentum running average coefficient
                    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

            #return loss
            return updates  # Return updates for all parameters


    # Credit : https://github.com/egg-west/AdamW-pytorch/blob/master/adamW.py
    class AdamW(Optimizer):
        """Implements Adam algorithm.

        It has been proposed in `Adam: A Method for Stochastic Optimization`_.

        Arguments:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float, optional): learning rate (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-8)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            amsgrad (boolean, optional): whether to use the AMSGrad variant of this
                algorithm from the paper `On the Convergence of Adam and Beyond`_

        .. _Adam\: A Method for Stochastic Optimization:
            https://arxiv.org/abs/1412.6980
        .. _On the Convergence of Adam and Beyond:
            https://openreview.net/forum?id=ryQu7f-RZ
        """

        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                    weight_decay=0, amsgrad=False):
            if not 0.0 <= lr:
                raise ValueError("Invalid learning rate: {}".format(lr))
            if not 0.0 <= eps:
                raise ValueError("Invalid epsilon value: {}".format(eps))
            if not 0.0 <= betas[0] < 1.0:
                raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
            if not 0.0 <= betas[1] < 1.0:
                raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
            defaults = dict(lr=lr, betas=betas, eps=eps,
                            weight_decay=weight_decay, amsgrad=amsgrad)
            super(AdamW, self).__init__(params, defaults)

        def __setstate__(self, state):
            super(AdamW, self).__setstate__(state)
            for group in self.param_groups:
                group.setdefault('amsgrad', False)

        def step(self, closure=None):
            """Performs a single optimization step.

            Arguments:
                closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
            """
            loss = None
            if closure is not None:
                loss = closure()

            step_sizes = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        # print("p.grad is None")
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    amsgrad = group['amsgrad']

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p.data)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                        if amsgrad:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    if amsgrad:
                        max_exp_avg_sq = state['max_exp_avg_sq']
                    beta1, beta2 = group['betas']

                    state['step'] += 1

                    # if group['weight_decay'] != 0:
                    #     grad = grad.add(group['weight_decay'], p.data)

                    # Decay the first and second moment running average coefficient
                    # exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    if amsgrad:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                    else:
                        denom = exp_avg_sq.sqrt().add_(group['eps'])

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                    step_sizes.append(step_size)

                    # p.data.addcdiv_(-step_size, exp_avg, denom)
                    # p.data.add_(-step_size,  torch.mul(p.data, group['weight_decay']).addcdiv_(1, exp_avg, denom) )
                    p.data.addcdiv_(torch.mul(p.data, group['weight_decay']), denom, value=-step_size).add_(exp_avg, alpha=-step_size)

            #return loss
            return step_sizes


    class AdamW_on_Lion_Optimizer(Optimizer):
        def __init__(self, params, lr=1e-3, adam_betas=(0.9, 0.999),
                     lion_betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

            self.params = list(params)

            # Define the Adam and Lion optimizers
            self.adamW = AdamW(self.params, lr=lr, betas=adam_betas,
                                     eps=eps, weight_decay=weight_decay)
            self.lion = Lion(self.params, lr=lr, betas=lion_betas,
                             weight_decay=weight_decay)

            self.scheduler_adamW = CosineAnnealingWarmRestarts(self.adamW, T_0=5, T_mult=2)
            self.scheduler_lion = CosineAnnealingWarmRestarts(self.lion, T_0=5, T_mult=2)

            defaults = dict(lr=lr, adam_betas=adam_betas,
                            lion_betas=lion_betas, eps=eps,
                            weight_decay=weight_decay)
            super().__init__(self.params, defaults)

        def get_current_lr(self, optimizer):
            """Retrieves the current learning rate, considering potential schedulers.
            """
            # Typically, the learning rate is stored in the first param_group
            # assuming all param_groups have the same lr if they exist
            return optimizer.param_groups[0]['lr']

        def step(self, lr=1e-3, max_iter=25, closure=None):
            """Performs a single optimization step.

            Args:
                closure (callable, optional): A closure that reevaluates
                                            the model and returns the loss.

            Returns:
              the loss.
            """
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            # Retrieve current learning rates from the optimizers
            lr_adamW = self.get_current_lr(self.adamW)
            lr_lion = self.get_current_lr(self.lion)

            for i in range(max_iter):
                # Apply the Lion and Adam optimizer
                lion_updates = self.lion.step()
                adamW_updates = self.adamW.step()

                scaled_updates = []
                for lion_update, adamW_update in zip(lion_updates, adamW_updates):
                    # Implement scaling logic with individual 'lion_update' and 'adamw_update'

                    # See [Learning Rate Grafting Transferability of Optimizer Tuning]
                    # (https://openreview.net/forum?id=FpKgG31Z_i9)
                    # Grafting adamW#lion: update direction from lion, update magnitude from adamW
                    # scaled_update = lion_step * (adamW_norms / lion_norms)
                    # Incorporate learning rates into the scaling factor (lr_adamW / lr_lion)
                    scaled_update = (lr_adamW / lr_lion) * \
                        (lion_update + 1e-10) * np.linalg.norm(adamW_update) / (np.linalg.norm(lion_update) + 1e-10)

                    scaled_updates.append(scaled_update)

                    # print(f"i = {i} , lion_update = {lion_update} , adamW_update = {adamW_update}, \
                    #      scaled_update = {scaled_update}")

                # Update model weights
                for param, update in zip(self.params, scaled_updates):
                    param.data.add_(update, alpha=-self.defaults['lr'])

                # Step the schedulers
                self.scheduler_adamW.step()
                self.scheduler_lion.step()
                # self.scheduler_grafted.step()  # If using a schedule for the grafted optimizer

            return scaled_updates

    optimizer_current_policy = AdamW_on_Lion_Optimizer(
                                    params=current_policy.parameters(),
                                    lr=1e-3,
                                    adam_betas=(0.9, 0.999),
                                    lion_betas=(0.9, 0.999),
                                    eps=1e-8,
                                    weight_decay=0
                               )
    optimizer_reference_policy = AdamW_on_Lion_Optimizer(
                                    params=reference_policy.parameters(),
                                    lr=1e-3
                                 )

elif USE_ADAMW:
    optimizer_current_policy = optim.AdamW(current_policy.parameters(), lr=1e-3)
    optimizer_reference_policy = optim.AdamW(reference_policy.parameters(),
                                             lr=1e-3)

# Training loop
num_epochs = 25  # Number of epochs to train for

for epoch in tqdm(range(num_epochs)):  # loop over the dataset multiple times

    """
        train_loader is an iterable of states that we're training on.
        action_space is the set of all possible actions.
        human_preferences is a dictionary or function that provides
        human preference scores.
        learning_rate is the learning rate η.
    """

    # current_policy: current policy network
    # reference_policy: reference policy network
    # state: current state from the environment

    total_loss = 0  # this variable is only valid for one epoch

    for batch in train_loader:
        optimizer_current_policy.zero_grad()
        optimizer_reference_policy.zero_grad()

        state = batch['state'].clone().to(device)
        state_action_a = batch['state_action_a'].clone().to(device)
        state_action_b = batch['state_action_b'].clone().to(device)
        mask_a = batch['mask_a'].clone().to(device)
        mask_b = batch['mask_b'].clone().to(device)
        human_preferences = batch['preference'].clone().to(device)

        """
        See Section 7.1 of the NLHF paper.

        The current policy response and the alternative policy response are
        generated token by token. For the alternative policy, the marginal
        geometric mixture is computed for each token based on the current
        policy probabilities and the reference policy probabilities.
        The alternative token is then sampled from this marginal mixture
        probability distribution.

        In the context of language models, generating responses often
        involves an autoregressive process where tokens are generated
        sequentially, conditioned on the previous tokens. The
        one-step-at-a-time regularized policy addresses this scenario
        by generating the alternative response token by token, rather
        than generating the entire response at once.

        The key idea behind the one-step-at-a-time regularized policy is
        to use a marginal geometric mixture of the current policy and the
        reference policy for each token generation step. This means that
        at each step, the probability distribution over the next token is
        computed as a geometric mixture of the probabilities assigned by
        the current policy and the reference policy.

        Mathematically, for each token position 𝑛, the marginal geometric
        mixture is defined as:

        log ˜𝜋𝜏_𝜃(𝑦_𝑛 | 𝑥, 𝑦_0:𝑛−1) = 𝜏 log 𝜋_𝜃(𝑦_𝑛 | 𝑥, 𝑦_0:𝑛−1) +
                                      (1 − 𝜏) log 𝜇(𝑦_𝑛 | 𝑥, 𝑦_0:𝑛−1) +
                                      𝐶(𝑥, 𝑦_0:𝑛−1)

        Here, ˜𝜋𝜏_𝜃 represents the one-step-at-a-time regularized policy,
        𝜋_𝜃 is the current policy, 𝜇 is the reference policy,
        𝜏 is the mixing coefficient, and
        𝐶(𝑥, 𝑦_0:𝑛−1) is a normalization constant that depends on the
        context 𝑥 and the generated tokens up to position 𝑛−1.

        To sample from this marginal geometric mixture, the logits of
        both the current policy and the reference policy are evaluated,
        conditioned on the context and the previously generated tokens.
        The logits are then combined using an arithmetic mixture
        weighted by 𝜏, and the next token is sampled from the resulting
        softmax distribution.

        The one-step-at-a-time regularized policy has a few important
        implications:

        - It allows for a more fine-grained control over the generation
          process, as the mixture between the current policy and the
          reference policy is applied at each token generation step.
        - It provides a way to incorporate the reference policy's
          influence throughout the generation process, rather than just
          at the beginning.
        - The normalization constant 𝐶(𝑥, 𝑦_0:𝑛−1) ensures that the
          resulting probability distribution over tokens is properly
          normalized at each step.

        However, it's important to note that the one-step-at-a-time
        regularized policy is an approximation to the full regularized
        policy defined in the paper. The full regularized policy would
        involve a geometric mixture over the entire sequence of tokens,
        which can be computationally challenging to compute and sample
        from directly.

        Implementing the one-step-at-a-time regularized policy in the
        Nash-MD algorithm requires modifying the response generation
        process to generate tokens sequentially and computing the
        marginal geometric mixture at each step. This allows for a
        more faithful implementation of the regularized policy while
        still being computationally tractable.

        Credit: Claude-3-Opus-200k AI chatbot
        """

        """
        To generate the alternative policy response, we follow a
        similar process but with a key difference. We initialize
        an empty list alternative_response and create a copy of
        the state_action tensor called alternative_state_action.
        Then, for each token generation step:
        - We obtain the current policy probabilities using
          current_policy(current_state_action, mask).
        - We obtain the reference policy probabilities using
          reference_policy(alternative_state_action, mask).
        - We compute the marginal geometric mixture of the current
          policy probabilities and reference policy probabilities
          using the formula:
            log_marginal_mixture =
                    tau * torch.log(current_policy_prob) +
                    (1 - tau) * torch.log(reference_policy_prob).
        - We sample a token from the marginal mixture probabilities
          using torch.multinomial() and append it to
          alternative_response.
        - We update alternative_state_action by concatenating the
          generated token to it, which will be used as input for
          the next token generation step.
        """

        # Calculate the average response length in the dataset
        response_lengths = [len(entry['human_ref_A'].split()) + len(entry['human_ref_B'].split())
                            for entry in dataset['train']]
        avg_response_length = sum(response_lengths) / len(response_lengths)

        # Set max_response_length to a value slightly higher than the average
        max_response_length = int(avg_response_length * 1.2)

        # Generate current policy response token by token
        current_state_action = state_action_a  # Use state_action_a as the initial state-action pair for the current policy
        current_response = []

        # Generate alternative policy response token by token
        alternative_state_action = state_action_b  # Use state_action_b as the initial state-action pair for the alternative policy
        alternative_response = []

        combined_loss = 0
        for _ in range(max_response_length):  # Loop over every tokens in the sequence
            current_policy_prob = current_policy(current_state_action)
            reference_policy_prob = reference_policy(alternative_state_action)

            assert current_policy_prob >= 0
            assert current_policy_prob <= 1
            assert reference_policy_prob >= 0
            assert reference_policy_prob <= 1

            current_token = torch.multinomial(current_policy_prob, num_samples=1)
            current_response.append(current_token)
            current_state_action = torch.cat((current_token, current_state_action[:, 1:]), dim=1)

            log_marginal_mixture = tau * torch.log(current_policy_prob) + (1 - tau) * torch.log(reference_policy_prob)
            alternative_policy_prob = torch.exp(log_marginal_mixture)

            alternative_token = torch.multinomial(alternative_policy_prob, num_samples=1)
            alternative_response.append(alternative_token)
            alternative_state_action = torch.cat((alternative_token, alternative_state_action[:, 1:]), dim=1)


            # Calculate the preference losses
            current_preference_loss, alternative_preference_loss = preference_model(
                                   state,
                                   current_state_action,  # state_action_a,
                                   alternative_state_action,  # state_action_b,
                                   None,  # mask_a,
                                   None,  # mask_b,
                                   current_policy,
                                   reference_policy,
                                   human_preferences
                               )

            # Calculate the combined loss for the current token by linearly combining the
            # preference losses for both the current policy and the alternative policy
            token_combined_loss = alpha * current_preference_loss + (1 - alpha) * alternative_preference_loss

            # Accumulate the combined loss for the current token
            combined_loss += token_combined_loss

        # Calculate the average combined loss across all tokens
        combined_loss /= max_response_length

        """
        Theorem 1 in the paper is related to the convergence properties of the
        Nash-MD algorithm. It states that if we have a Nash equilibrium π*
        for the regularized preference model,
        the KL divergence between π* and the policy obtained at each iteration
        of the Nash-MD algorithm (π_t+1) is non-increasing and converges at a
        rate proportional to 1/sqrt(T), where T is the number of iterations.

        The convergence rate is affected by the choice of the learning rate η,
        which is suggested to be set as log(T)/T. This rate is significant
        because it dictates how quickly the policies converge to the Nash
        equilibrium in terms of KL divergence, a measure of the difference
        between two probability distributions.

        The theorem is crucial for understanding the Nash-MD loss function
        because it provides the theoretical foundation that guarantees the
        algorithm's policies will converge to a Nash equilibrium. The loss
        function used in the Nash-MD algorithm is designed to both maximize
        alignment with human preferences (as captured by the preference model)
        and minimize the KL divergence from the previous policy, which ensures
        that the updated policy does not diverge too rapidly from the previous
        iterations.

        This careful balance allows sequence of policies to improve steadily
        while maintaining a trajectory toward the Nash equilibrium.
        """

        """
        In equation (4), the KL divergence serves as a regularization term
        within the arg max operation, but once we solve for π_t+1 optimization
        problem, its effects are embedded in the form of the solution
        in equation (5) and don't need to be listed separately.

        # Calculate the KL divergence part of the Nash-MD objective
        KL_divergence = torch.distributions.kl_divergence(
            torch.distributions.Categorical(probs),
            torch.distributions.Categorical(normalized_probs)
        )
        """

        # Perform backpropagation
        combined_loss.backward(retain_graph=True)

        # Clip gradients: gradients are modified in place
        max_grad_norm = 1.5
        for model in [current_policy, reference_policy]:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            for name, param in model.named_parameters():
                if 'out_proj.bias' not in name:
                    # clip weights but not bias for out_proj
                    torch.nn.utils.clip_grad_norm_(param,
                                                   max_norm=max_grad_norm)

        if debugging_is_on:
            print("DEBUGGING IS ON !!!")

            for model in [current_policy, reference_policy]:
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        print(f"{name} gradient: \
                              {parameter.grad.data.norm(2)}")
                    else:
                        print(f"{name} has no gradient")

        optimizer_current_policy.step()
        optimizer_reference_policy.step()

        total_loss += combined_loss.item()

    train_loss = total_loss / len(train_loader)

    if not train_loss >= 0:
        print("non-positive training loss !!!")
        debugging_is_on = True

    print(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}')
