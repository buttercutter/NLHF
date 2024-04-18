# [NLHF : Nash Learning from Human Feedback](http://arxiv.org/abs/2312.00886)
# [IPO-MD : Human Alignment of Large Language Models through Online Preference Optimisation](http://arxiv.org/abs/2403.08635)


import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, LambdaLR
from tqdm import tqdm


##########################################################################################

USE_ATARI = 0
USE_NLP = ~USE_ATARI

if USE_NLP:
    # only enables one of the following
    USE_BERT = 1
    USE_CAUSAL_LM = 0
    USE_SEQ2SEQ_LM = 0
    USE_MAMBA = 0

USE_ADAMW_ON_LION = 0
USE_ADAMW = ~USE_ADAMW_ON_LION

USE_IPO_MD = 0
USE_NLHF = ~USE_IPO_MD

##########################################################################################


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


# learning rate
# result is the best with beta=3e-5 according to Appendix B.3 of IPO-MD paper
if USE_IPO_MD:
    lr = 1e-4
elif USE_NLHF:
    lr = 3e-5

# Adjusting beta affects the balance between exploiting the current policy
# and exploring new policies suggested by the reference model or feedback.
# result is the best with beta=0.125 according to Appendix B.3 of IPO-MD paper
beta = 0.125

# temperature parameter controlling degree of kl regularisation towards π_ref,
# independent of beta parameter above used for mixture
# result is the best with tau=0.008 according to Appendix B.3 of IPO-MD paper
if USE_IPO_MD:
    tau = 1.0
elif USE_NLHF:
    tau = 0.008

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
        # bert_model = AutoModel.from_pretrained("prajjwal1/bert-medium").to(device)
    elif USE_CAUSAL_LM:
        # bert_model = AutoModelForCausalLM.from_pretrained("ContextualAI/archangel_sft-dpo_llama7b")
        bert_model = AutoModelForCausalLM.from_pretrained("AdamG012/chat-opt-350m-reward-deepspeed")
    elif USE_MAMBA:
        bert_model = AutoModelForCausalLM.from_pretrained("Q-bert/Mamba-370M", trust_remote_code=True)
    elif USE_SEQ2SEQ_LM:
        bert_model = AutoModelForSeq2SeqLM.from_pretrained("stanfordnlp/SteamSHP-flan-t5-large")

    # for attr, value in bert_model.config.__dict__.items():
    #     print(f"{attr}: {value}")

    if USE_MAMBA:
        bert_model.config.hidden_size = bert_model.config.d_inner

    print(f"bert_model.config.hidden_size = {bert_model.config.hidden_size}")

    class PolicyNetwork(nn.Module):
        def __init__(self, state_dim):
            super(PolicyNetwork, self).__init__()
            self.bert = bert_model
            self.final_layer = nn.Linear(self.bert.config.hidden_size, num_actions)
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
            possible_next_tokens = self.final_layer(text_embeddings)

            # for sampling next token y_n distributions (implemented as a softmax over logits)
            # See section 7.1 of NLHF paper for more details
            possible_next_tokens = torch.softmax(possible_next_tokens, dim=-1)

            return possible_next_tokens


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
    log_alternative_policy_probs = (1 - beta) * torch.log(current_policy_probs) + \
                                beta * torch.log(reference_policy_probs)

    alternative_policy_probs = torch.exp(log_alternative_policy_probs)

    """
    See section 7 of the paper

    In equation (11), the normalization constant c is indeed important.
    It ensures that after updating the policy using the Nash-MD-PG algorithm,
    the updated policy π is still a valid probability distribution.

    The constant c is determined after the update so that the
    sum of probabilities across all possible actions y equals 1.
    """
    # Normalize the alternative policy 𝜋′ using equation (11)
    alternative_policy_probs = alternative_policy_probs / alternative_policy_probs.sum(dim=-1, keepdim=True)

    """
    The following needs some pondering and may need to be revisited.

    Notice that the one-step-at-a-time regularized policy 𝜋𝜏 𝜃(𝑦|𝑥)
    is different from the original regularized policy 𝜋𝜏 𝜃(𝑦|𝑥)
    because the sequence of normalization constants 𝐶(𝑥, 𝑦0:𝑛−1)
    depend on the specific sample path 𝑦0:𝑛−1 and does not necessarily
    correspond to the full normalization constant 𝑐 defined in Equation (5).

    Credit: Quoted directly from section 7.1 of NLHF paper
    """

    assert torch.all(alternative_policy_probs >= 0)
    assert torch.all(alternative_policy_probs <= 1)

    return alternative_policy_probs


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

    baseline = 0.5 is subtracted to center the preference around the middle.
    Subtract the baseline (average preference score) for variance reduction
    to reduce the variance of the policy gradient estimate, which can help
    stabilize training
    we do not use preference_score.mean(), see equation (9) in NLHF paper

    There is also another reason for using baseline = 0.5, see
    proposition D.1 in the IPO-MD paper:

    p(y ≻ y′) = 1 - p(y′ ≻ y) for all y, y′ ∈ Y
    f(y, y′) := p(y ≻ y′) - 1/2 - τ log(π(y)/π_ref(y)) + τ log(π(y′)/π_ref(y′))

    So, the baseline is set to 1/2 to ensure that the preference loss f()
    satisfies the requirements of f(y, y′) = -f(y′, y)

    tau * kl_divergence is the KL divergence regularization term
    for state-action pair, weighted by the coefficient tau.

    So equation (9) effectively maximizes the probability of the
    preferred state-action pair while minimizing the probability
    of the less preferred pair.

    It encourages the policy to stay close to the reference policy
    for the state-action pair.
    '''
    # for Stanford SHP dataset, it is 0.5 because human preference label is either 0 or 1
    baseline = 0.5

    # Calculate the preference loss using the cross-entropy losses as well as
    # the expectation over states and actions using equation (10)
    # We remove `baseline` from both `preference_loss` and `preference` as follows because
    # the mean or average of `human_preferences` and `(1 - human_preferences)` which gives 1/2
    # already emulates the effect of subtracting `baseline = 1/2`

    # preference_loss = torch.mean(((human_preferences - baseline - tau * current_kl_estimator_a) * current_gradient_term_a +
    #                              ((1 - human_preferences) - baseline - tau * current_kl_estimator_b) * current_gradient_term_b)) / 2)
    current_preference_loss = torch.mean(((human_preferences - tau * current_kl_estimator_a) * current_loss_a +
                                 ((1 - human_preferences) - tau * current_kl_estimator_b) * current_loss_b) / 2)

    current_preference = ((human_preferences - tau * current_kl_estimator_a) + \
                            ((1 - human_preferences) - tau * current_kl_estimator_b)) / 2


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
    alternative_kl_divergence_a = log_current_prob_a - log_alternative_prob_a
    alternative_kl_divergence_b = log_current_prob_b - log_alternative_prob_b

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
    # We remove `baseline` from both `preference_loss` and `preference` as follows because
    # the mean or average of `human_preferences` and `(1 - human_preferences)` which gives 1/2
    # already emulates the effect of subtracting `baseline = 1/2`

    # alternative_preference_loss = torch.mean(((human_preferences - baseline - tau * alternative_kl_estimator_a) * alternative_gradient_term_a +
    #                                          ((1 - human_preferences) - baseline - tau * alternative_kl_estimator_b) * alternative_gradient_term_b)) / 2)
    alternative_preference_loss = torch.mean(((human_preferences - tau * alternative_kl_estimator_a) * alternative_loss_a +
                                             ((1 - human_preferences) - tau * alternative_kl_estimator_b) * alternative_loss_b) / 2)

    alternative_preference = ((human_preferences - tau * alternative_kl_estimator_a) + \
                                ((1 - human_preferences) - tau * alternative_kl_estimator_b)) / 2
    
    assert torch.all(current_preference <= 1)
    assert torch.all(alternative_preference <= 1)

    return current_preference_loss, alternative_preference_loss, current_preference, alternative_preference


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

    # For Atari games, get the action space size from the environment
    num_actions = env.action_space.n

    state = env.reset()  # Reset the environment to get the initial state

    # Assuming the first element of the tuple is the screen image we want
    screen = state[0] if isinstance(state, tuple) else state

    # Add batch dimension
    state_tensor = torch.tensor(screen, dtype=torch.float32).unsqueeze(0)
    state_size = 33600


elif USE_NLP:

    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Load all the data from https://huggingface.co/datasets/stanfordnlp/SHP
    # dataset = load_dataset("stanfordnlp/shp")

    # Load one of the subreddits
    dataset = load_dataset("stanfordnlp/shp", data_dir="explainlikeimfive")

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # tokenizer = AutoTokenizer.from_pretrained("ContextualAI/archangel_sft-dpo_llama7b")
    # tokenizer = AutoTokenizer.from_pretrained("AdamG012/chat-opt-350m-reward-deepspeed")
    # tokenizer = AutoTokenizer.from_pretrained("stanfordnlp/SteamSHP-flan-t5-large")

    # For NLP models,  get the action space size from tokenizer's vocab size
    num_actions = tokenizer.vocab_size

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
# This matches NLHF equation (10) stop-gradient requirement for `alternative_policy`
# where it is being derived from both `current_policy` and `reference_policy`
# using equation (11).
# Quoted from [IPO-MD paper](http://arxiv.org/abs/2403.08635):
# Nash-MD-PG is on-policy, in that the only gradient contributions appearing in
# its update are those corresponding to actions sampled under the current policy.
if USE_IPO_MD:
    current_policy.eval()
elif USE_NLHF:
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
                                    lr=lr
                               )
    optimizer_reference_policy = AdamW_on_Lion_Optimizer(
                                    params=reference_policy.parameters(),
                                    lr=lr
                                 )

elif USE_ADAMW:
    optimizer_current_policy = optim.AdamW(current_policy.parameters(), lr=lr)
    optimizer_reference_policy = optim.AdamW(reference_policy.parameters(), lr=lr)


"""
Lemma 1 in the NLHF paper is a general statement about the relationship between
the KL divergences of different policies. It does not specifically depend on the
representation of the policies as tensors or scalars.

In the context of the NLHF paper, the policies are typically represented as
probability distributions over actions or tokens.

These probability distributions can be represented in different ways,
depending on the specific problem and the chosen policy parameterization.

1. Tensor representation: In many cases, policies are represented as tensors,
   especially when dealing with complex action spaces or sequences.
   For example, in language models, the policies are often represented as
   tensors of probabilities over the vocabulary at each step, as discussed in
   the previous response.

2. Scalar representation: In simpler cases, policies can be represented as scalars.
   For example, if the action space is binary (e.g., a yes/no decision), the policy
   could be represented as a single scalar value indicating the probability of
   taking one of the actions.

Lemma 1 holds regardless of the specific representation of the policies, as long as
the KL divergences between the policies can be computed. Lemma 1 states that:

KL(π, πₜ^μ) ≤ ητKL(π, μ) + (1 - ητ)KL(π, πₜ)

where:
- π is any policy
- πₜ is the policy at iteration t
- μ is the reference policy
- πₜ^μ is the policy obtained by a geometric mixture of πₜ and μ
- ητ is a scalar value between 0 and 1

The KL divergences in the Lemma 1, such as KL(π, μ) and KL(π, πₜ), are scalar values
that measure the difference between two probability distributions.

These KL divergences can be computed regardless of whether the policies are
represented as tensors or scalars, as long as the probability distributions are
well-defined.

In summary, while the NLHF paper often uses tensor representations for policies
due to the nature of the problems being addressed, Lemma 1 itself is a general
statement that holds for any valid representation of policies, including scalars,
as long as the KL divergences can be computed.

Credit: Claude-3-Opus-200k
"""

# Compute step by step the KL divergence between two distributions (pi_1 and pi_2),
# with only single action or token at any compute cycle
def compute_kl_divergence(pi_1, pi_2):
    # Removes the unnessary dimension due to tokenizer() function
    # because torch.distributions.kl_divergence() needs more than 1 dimension
    pi_1 = pi_1.squeeze(0)
    pi_2 = pi_2.squeeze(0)

    # print(f"pi_1.shape = {pi_1.shape}, pi_2.shape = {pi_2.shape}, \
    #       num_actions = {num_actions}")

    kl_divergence = torch.distributions.kl_divergence(
        torch.distributions.Categorical(probs=pi_1),
        torch.distributions.Categorical(probs=pi_2)
    )
    return kl_divergence

# Generate a random policy 𝜋 for testing
def generate_random_policy(num_actions):
    # Generate a random tensor of probabilities for each action
    # at each step or sequence slot
    policy_probs = torch.rand((1, num_actions))
    # Softmax distribution required in section 7.1 of NLHF paper
    policy_probs = torch.softmax(policy_probs, dim=-1)
    return policy_probs
    
def test_lemma1(pi_t, pi_mu_t, mu, eta=3e-5, tau=tau):
    """
    This function asserts the inequality from Lemma 1 in the NLHF paper

    pi_t: policy at iteration t
    pi_mu_t: policy obtained by a geometric mixture of pi_t and mu
    mu: reference policy
    eta: learning rate, scalar value between 0 and 1
    tau: regularization coefficient, scalar value between 0 and 1
    sequence_length: length of the sequence
    """

    # Generates probability distribution for all different possible tokens
    # at each step or sequence slot
    pi = generate_random_policy(num_actions)  # Arbitrary policy 𝜋

    # Print out the policy to see the values
    # print(f"Arbitrary policy pi: {pi}")


    # Compute the KL divergences
    kl_pi_mu = compute_kl_divergence(pi, mu)
    kl_pi_pi_t = compute_kl_divergence(pi, pi_t)
    kl_pi_mu_t_mu = compute_kl_divergence(pi_mu_t, mu)

    # Calculate the right side of the inequality
    rhs = eta * tau * kl_pi_mu + (1 - eta * tau) * kl_pi_pi_t - eta * tau * kl_pi_mu_t_mu

    # Assert the inequality from Lemma 1
    assert compute_kl_divergence(pi, pi_mu_t) <= rhs, \
        "Lemma 1 inequality does not hold"

    return True  # If Lemma 1 passes, return True


# Please see Lemma 2 inside the NLHF paper
"""
Given the Stanford Human Preferences (SHP) dataset's nature, which typically
contains comparative preference data (e.g., user preferences between pairs of
options rather than explicit probabilities), and fixed probabilities are not 
directly provided. Therefore, the most suitable approach is dynamically computing 
preferences based on the context and current policy evaluations.

### Why Dynamic Preference Modeling Fits SHP Data:

1. **Relative Preferences**:
   - The SHP dataset usually includes paired comparisons (e.g., "Is option A
   better than option B?"). These can be used to infer relative strengths or
   preferences but do not directly give the probability P(y > pi_mu_t)
   for action y over a policy pi_mu_t.

2. **Learning from Feedback**:
   - Preferences can be modeled using a function that interprets the outcomes of
   these comparisons. For instance, a neural network or a logistic regression model
   that inputs features of y and pi_mu_t and outputs the likelihood of y being preferred.
   This model adapts and improves as it learns from more data.

3. **Adaptability**:
   - Dynamically computed preferences allow the system to adapt to new data and
   potentially changing user preferences, which is crucial in real-world applications
   like recommendation systems where user preferences aren't static.

### Implementing a Dynamic Preference Model

Given the paired nature of data in SHP, here's how we could set up a dynamic
preference computation:

1. **Model Definition**:
   - Define a model that predicts the preference probability between two given options.
   This could be a logistic regression model or a neural network.

2. **Training the Model**:
   - Use the paired comparisons to train this model. For each pair (A, B) and a
   user preference P(A > B), the model learns to predict the likelihood of
   preference for one option over the other.

3. **Using the Model for Policy Evaluation**:
   - When evaluating P(y > pi_mu_t), use the trained model to compute the
   preference probability of each action y over the actions suggested by pi_mu_t.


### Conclusion
Using a dynamic preference model is particularly beneficial for datasets like SHP,
where preferences are context-dependent and relative. This approach is more flexible
and generally more robust to changes in data distribution or user behavior over time.
The key is to ensure that the preference model is well-calibrated and accurately
reflects the nuances of human preferences as captured in the dataset.

Credit: GPT4
"""
def test_lemma2(pi, pi_t_plus_1, pi_mu_t, eta, alternative_preference):
    """
    Validate the inequality stated in Lemma 2 Equation (14) of the NLHF paper.
    We do not use the final expression of Lemma 2 because it is not directly
    expressible in code, given that the Nash policy (𝜋∗𝜏) is not feasible to obtain
    for all different kinds of downstream tasks.

    Instead, we use intermediate step which is equation (14) to validate Lemma 2.
    The final expression of Lemma 2 makes use of Lemma 1, but since we had already
    tested Lemma 1 above, we can directly use the intermediate step of Lemma 2.

    Args:
    - pi: Torch tensor, any policies.
    - pi_t_plus_1: Torch tensor, the policy at time t+1.
    - pi_mu_t: Torch tensor, the alternative mixture policy at time t.
    - eta: float, the learning rate or step size.
    - alternative_preference: Torch tensor, the alternative_preference probabilities P(y > 𝜋𝛽𝜃) for all y.
    - current_preference: Torch tensor, the current_preference probabilities P(y > 𝜇) for all y.
                          See section 7.3 on why we can replace P(y > 𝜋𝛽𝜃) with P(y > 𝜇)
    """

    if pi_mu_t is None:
        # t=0, just started token generation of the very first token, and there is no policy yet for
        # the KL divergence comparison between timestep = t and t+1, so no meaningful testing result
        # given that we are feeding in "previous" mixture_policy and "current" policy as pi_mu_t and 
        # pi_t_plus_1 respectively
        print("At initial timestep, no policies available for comparison. Skipping Lemma 2 test...")
        return True  # If policies are None, return True

    # Generates probability distribution for all different possible tokens
    # at each step or sequence slot
    pi = generate_random_policy(num_actions)  # Arbitrary policy 𝜋

    # Compute the KL divergences
    KL_pi_pi_t_plus_1 = compute_kl_divergence(pi, pi_t_plus_1)
    KL_pi_pi_mu_t = compute_kl_divergence(pi, pi_mu_t)

    # Calculate the preference sum
    """
    In equation (14), P(𝑦 ≻ 𝜋𝜇𝑡) represents the preference of action 𝑦 over the regularized policy 𝜋𝜇𝑡.
    This notation is defined earlier in the paper in Section 3, where P(𝑦 ≻ 𝜋′|𝑥) = 𝔼𝑦′∼𝜋′(·|𝑥)[P(𝑦 ≻ 𝑦′|𝑥)]
    is the preference of an action 𝑦 over a policy 𝜋′ conditioned on context 𝑥.

    The contextual bandit setup described at the end of Section 6 is the most relevant here.
    It defines the Nash-MD algorithm for the contextual case, where for every 𝑥 ∈ supp(𝜌):

    𝜋𝑡+1(·|𝑥) = arg max𝜋(·) [𝜂P(𝜋(·|𝑥) ≻ 𝜋𝜇𝑡(·|𝑥)|𝑥) − KL(𝜋(·), 𝜋𝜇𝑡(·|𝑥))]

    and 𝜋𝜇𝑡(𝑦|𝑥) ∝ 𝜋𝑡(𝑦|𝑥)1−𝜂𝜏𝜇(𝑦|𝑥)𝜂𝜏.

    So in the proof of Theorem 1, when applying Lemma 2 to the contextual case, 𝛿(𝑦) = 𝜂P(𝑦 ≻ 𝜋𝜇𝑡)
    uses the preference of action 𝑦 over the regularized policy 𝜋𝜇𝑡.

    In contrast, equation (9) describes the policy gradient estimate used in the model-based approach
    for computing the Nash equilibrium. It uses the reference policy 𝜇(𝑦|𝑥) directly in the KL term,
    rather than the regularized policy 𝜋𝜇𝑡(𝑦|𝑥).

    Mathematically, equation (9) and the contextual bandit setup are related through the discussion
    in Section 7.3 comparing Nash-MD and Nash-MD-PG. It's shown there that regularizing with respect
    to 𝜋𝜇𝑡 (as in Nash-MD) is equivalent to regularizing with respect to 𝜇 (as in Nash-MD-PG), because:

    KL(𝜋𝜃, 𝜋𝛽𝜃) = 𝛽KL(𝜋𝜃, 𝜇) − 𝔼𝑥∼𝜌[𝑐(𝑥)]

    and so ∇𝜃KL(𝜋𝜃, 𝜋𝛽𝜃) = 𝛽∇𝜃KL(𝜋𝜃, 𝜇).

    Since Nash-MD-PG performs a single gradient step before updating 𝜋𝜃, using 𝜋𝛽𝜃 or 𝜇 in the
    regularization is equivalent. The additional 𝛽 parameter in equation (11) defining 𝜋𝛽𝜃 allows
    it to be tuned independently of 𝜏 used in equation (9).

    Let's derive the equivalence of using 𝜋𝛽𝜃 or 𝜇 in the regularization term of the policy gradient estimate.
    In section 7.3 , we have :

    KL(𝜋𝜃, 𝜋 𝛽 𝜃) = (1 − 𝛽)KL(𝜋𝜃, 𝜋𝜃) + 𝛽KL(𝜋𝜃, 𝜇) − 𝔼𝑥∼𝜌 [𝑐(𝑥)] = 𝛽KL(𝜋𝜃, 𝜇) − 𝔼𝑥∼𝜌 [𝑐(𝑥)], 
    
    where 𝑐(𝑥) is the normalizing constant in Equation (11). Thus, we have ∇ 𝜃KL(𝜋𝜃, 𝜋 𝛽 𝜃) = 𝛽∇𝜃KL(𝜋𝜃, 𝜇)

    This expression relates the KL divergence between the current policy 𝜋𝜃 and the mixture policy 𝜋 𝛽 𝜃 to 
    the KL divergence between the current policy 𝜋𝜃 and the reference policy 𝜇. Let's break it down step by step:

    1. KL(𝜋𝜃, 𝜋 𝛽 𝜃) represents the KL divergence between the current policy 𝜋𝜃 and the mixture policy 𝜋 𝛽 𝜃.

    2. The mixture policy 𝜋 𝛽 𝜃 is defined as a geometric mixture of the current policy 𝜋𝜃 and the reference policy 𝜇, 
    with mixing coefficient 𝛽. Specifically, 𝜋 𝛽 𝜃(𝑦|𝑥) = 𝜋𝜃(𝑦|𝑥)^(1−𝛽) 𝜇(𝑦|𝑥)^𝛽.

    3. The KL divergence between the current policy and the mixture policy can be decomposed into two terms:
    - (1 − 𝛽)KL(𝜋𝜃, 𝜋𝜃): This term represents the KL divergence between the current policy and itself, which is always zero.
    - 𝛽KL(𝜋𝜃, 𝜇): This term represents the KL divergence between the current policy and the reference policy, scaled by the mixing coefficient 𝛽.

    4. 𝔼𝑥∼𝜌 [𝑐(𝑥)] represents the expected value of the normalizing constant 𝑐(𝑥) over the state distribution 𝜌. 
    The normalizing constant 𝑐(𝑥) is used in Equation (11) to ensure that the updated policy probabilities sum up to 1.

    5. By subtracting the expected value of the normalizing constant 𝔼𝑥∼𝜌 [𝑐(𝑥)] from both sides of the equation, we obtain:
    KL(𝜋𝜃, 𝜋 𝛽 𝜃) = 𝛽KL(𝜋𝜃, 𝜇) − 𝔼𝑥∼𝜌 [𝑐(𝑥)]

    6. Finally, taking the gradient of both sides with respect to the policy parameters 𝜃 yields:
    ∇ 𝜃KL(𝜋𝜃, 𝜋 𝛽 𝜃) = 𝛽∇𝜃KL(𝜋𝜃, 𝜇)

    This equation states that the gradient of the KL divergence between the current policy and the mixture policy with 
    respect to the policy parameters is equal to 𝛽 times the gradient of the KL divergence between the current policy and the reference policy.

    The significance of this expression lies in the fact that it establishes a connection between the regularization term 
    used in Nash-MD (KL(𝜋𝜃, 𝜋 𝛽 𝜃)) and the regularization term used in Nash-MD-PG (KL(𝜋𝜃, 𝜇)). It shows that the gradient of 
    the KL divergence between the current policy and the mixture policy can be computed using only the gradient of the KL divergence 
    between the current policy and the reference policy, scaled by the mixing coefficient 𝛽.

    This relationship allows for a simplification of the regularization term in the Nash-MD-PG algorithm, as it can be expressed 
    solely in terms of the KL divergence between the current policy and the reference policy, without explicitly involving the mixture policy.

    Now, we will derive the maths inside section 7.3 to rigorously prove the above:

    Expand the logarithm of the mixture policy using the definition of 𝜋 𝛽 𝜃.
    𝜋𝛽𝜃(𝑦|𝑥) = (𝜋𝜃(𝑦|𝑥))^(1-𝛽) * (𝜇(𝑦|𝑥))^𝛽
    log 𝜋𝛽𝜃(𝑦|𝑥) = (1 − 𝛽) log(𝜋𝜃(𝑦|𝑥)) + 𝛽 log(𝜇(𝑦|𝑥)) + 𝑐(𝑥)

    Now, the KL divergence between 𝜋𝜃 and 𝜋𝛽𝜃 can be derived as follows:

    KL(𝜋𝜃, 𝜋𝛽𝜃) = ∑𝑦 𝜋𝜃(𝑦|𝑥) log(𝜋𝜃(𝑦|𝑥) / 𝜋𝛽𝜃(𝑦|𝑥))
    = ∑𝑦 𝜋𝜃(𝑦|𝑥) [log 𝜋𝜃(𝑦|𝑥) - log 𝜋𝛽𝜃(𝑦|𝑥)]
    = ∑𝑦 𝜋𝜃(𝑦|𝑥) [log 𝜋𝜃(𝑦|𝑥) - ((1-𝛽) log 𝜋𝜃(𝑦|𝑥) + 𝛽 log 𝜇(𝑦|𝑥) - log Z)]
    = 𝛽 ∑𝑦 𝜋𝜃(𝑦|𝑥) [log 𝜋𝜃(𝑦|𝑥) - log 𝜇(𝑦|𝑥)] + log Z
    = 𝛽 KL(𝜋𝜃, 𝜇) + log Z

    where Z = ∑𝑦 (𝜋𝜃(𝑦|𝑥))^(1-𝛽) * (𝜇(𝑦|𝑥))^𝛽 is the normalizing constant.

    Therefore, we have:

    ∇𝜃 KL(𝜋𝜃, 𝜋𝛽𝜃) = 𝛽 ∇𝜃 KL(𝜋𝜃, 𝜇)

    This shows that regularizing with respect to the mixture 𝜋𝛽𝜃 (in Nash-MD) is equivalent to regularizing w.r.t. 𝜇 (in Nash-MD-PG), 
    since a single gradient descent step is performed before updating 𝜋𝜃.

    Credit: Claude-3-Opus-200k
    """
    preference_sum = torch.sum((pi_mu_t - pi) * alternative_preference, dim=-1)  # sum over actions

    # Compute the RHS of the inequality
    rhs = KL_pi_pi_mu_t + eta * preference_sum + 2 * eta**2

    # Assert that the KL divergence of pi and pi_t_plus_1 is less than or equal to the calculated RHS
    assert KL_pi_pi_t_plus_1 <= rhs, "Lemma 2, Equation (14) does not hold."

    return True  # If Lemma 2 passes, return True


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
                    (1 - beta) * torch.log(current_policy_prob) +
                    beta * torch.log(reference_policy_prob).
        - We sample a token from the marginal mixture probabilities
          using torch.multinomial() and append it to
          alternative_response.
        - We update alternative_state_action by concatenating the
          generated token to it, which will be used as input for
          the next token generation step.
        """

        # Calculate the maximum response length in the batch
        # Instead of calculating the average response length across the entire dataset,
        # we calculate the maximum response length within the current batch.
        # We use the mask_a and mask_b tensors to determine the actual length of each response in the batch.
        # The mask_a and mask_b tensors have the same shape as state_action_a and state_action_b, respectively,
        # with values of 1 indicating valid tokens and 0 indicating padding tokens.
        # By applying sum(dim=1) to mask_a and mask_b, we obtain the actual length of each response in the batch.
        max_response_length = max(mask_a.sum(dim=1).max().item(), mask_b.sum(dim=1).max().item())

        # Generate current policy response token by token
        current_state_action = state_action_a  # Use state_action_a as the initial state-action pair for the current policy
        current_response = []

        # Generate alternative policy response token by token
        alternative_state_action = state_action_b  # Use state_action_b as the initial state-action pair for the alternative policy
        alternative_response = []

        # Initialize the log marginal mixture tensor and the combined loss
        log_marginal_mixture = torch.zeros((1, num_actions))
        combined_loss = 0

        # Initialize the previous policy probabilities for verification of the next token generation step
        current_policy_probs_previous_step = None
        alternative_policy_probs_previous_step = None

        for t in range(max_response_length):  # Loop over every token slot in the sequence
            # Obtains the current and reference policies probabilities
            # for all the possible actions/tokens in that specific slot
            current_policy_probs = current_policy(current_state_action)
            reference_policy_probs = reference_policy(alternative_state_action)

            assert torch.all(current_policy_probs >= 0)
            assert torch.all(current_policy_probs <= 1)
            assert torch.all(reference_policy_probs >= 0)
            assert torch.all(reference_policy_probs <= 1)

            # Samples a next token 𝑦𝑛 from the softmax distribution of current_policy
            # Implements token-per-token "autoregressive" (hence only up to previous t token) generation of responses 𝑦 ∼ 𝜋(·|𝑥)
            current_token = torch.multinomial(current_policy_probs[:, :t+1], num_samples=1)
            current_response.append(current_token)
            # print(f'current_token.size(): {current_token.size()}, current_state_action[:, 1:].size(): {current_state_action[:, 1:].size()}')
            current_state_action = torch.cat((current_token, current_state_action[:, 1:]), dim=-1)

            # See equation (11) of the NLHF paper
            # Calculate the marginal geometric mixture of the current policy and the reference policy
            log_marginal_mixture[:, :t+1] = (1 - beta) * torch.log(current_policy_probs[:, :t+1]) + beta * torch.log(reference_policy_probs[:, :t+1])
            alternative_policy_probs = torch.exp(log_marginal_mixture)

            assert torch.all(alternative_policy_probs >= 0)
            assert torch.all(alternative_policy_probs <= 1)

            # Samples a next token 𝑦𝑛 from the softmax distribution of alternative_policy
            alternative_token = torch.multinomial(alternative_policy_probs, num_samples=1)
            alternative_response.append(alternative_token)
            # print(f'alternative_token.size(): {alternative_token.size()}, alternative_state_action[:, 1:].size(): {alternative_state_action[:, 1:].size()}')
            alternative_state_action = torch.cat((alternative_token, alternative_state_action[:, 1:]), dim=-1)

            # Test Lemma 1 of the NLHF paper
            # print(f"current_policy_probs.shape = {current_policy_probs.shape}, \
            #        reference_policy_probs.shape = {reference_policy_probs.shape}, \
            #        alternative_policy_probs.shape = {alternative_policy_probs.shape}")
            test_lemma1(current_policy_probs, alternative_policy_probs, reference_policy_probs, eta=lr, tau=tau)

            # Calculate the preference losses
            current_preference_loss, alternative_preference_loss, current_preference, alternative_preference = preference_model(
                                   state,
                                   current_state_action,  # state_action_a,
                                   alternative_state_action,  # state_action_b,
                                   None,  # mask_a,
                                   None,  # mask_b,
                                   current_policy,
                                   reference_policy,
                                   human_preferences
                               )

            # Test Lemma 2 of the NLHF paper
            # We should test Lemma 2 for all different kinds of policy 𝜋 , as suggested in equation (14) description.
            # Previously, Lemma 2 test only passes using a specific policy 𝜋 = 𝜋_(t-1) , 
            # now removing `baseline` from the logic of `alternative_preference` resolved the issue of Lemma 2 test failure.
            test_lemma2(pi=current_policy_probs_previous_step, pi_t_plus_1=current_policy_probs,
                        pi_mu_t=alternative_policy_probs_previous_step, eta=lr, alternative_preference=alternative_preference)

            # Calculate the combined loss for the current token by linearly combining the
            # preference losses for both the current policy and the alternative policy
            token_combined_loss = alpha * current_preference_loss + (1 - alpha) * alternative_preference_loss

            # Accumulate the combined loss for the current token
            combined_loss += alternative_preference_loss #token_combined_loss

            # Update the previous policy probabilities for the next token generation step
            current_policy_probs_previous_step = current_policy_probs
            alternative_policy_probs_previous_step = alternative_policy_probs

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
