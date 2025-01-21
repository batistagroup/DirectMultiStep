# Mixture of Experts

This document describes the Mixture of Experts (MoE) components used in the DMS model. MoE is a technique that improves model capacity and efficiency by routing different inputs to specialized sub-networks (experts).

## Position-wise Feed-forward Layer

The standard feed-forward network serves as our baseline for comparison with MoE layers. It processes each position in the sequence independently through a simple two-layer network with expansion and projection. This is the traditional architecture used in transformer models.

## Noisy Top-k Router

The router is the brain of the MoE system - it decides which experts should process each token. Key features:

- Uses learned routing weights to match tokens with relevant experts
- Adds learned noise to encourage exploration and prevent expert collapse
- Selects top-k experts per token to enable specialization while maintaining redundancy
- Produces sparse routing probabilities to enable efficient computation

The noise mechanism is particularly important as it:

1. Prevents tokens from always taking the same path
2. Helps balance load across experts
3. Improves training stability

## Expert Network

Each expert is a specialized feed-forward network that becomes tuned to handle specific types of tokens or patterns. The expert architecture mirrors the standard feed-forward layer, but each expert can learn different specializations. For example:

- Some experts might focus on syntax
- Others on specific vocabulary domains
- Others on particular transformation patterns

## Sparse MoE Layer

This is where everything comes together into an efficient, scalable system:

1. **Token Routing**: The router examines each token and decides which experts should process it
2. **Load Balancing**:
    - Uses capacity factors to prevent expert overload
    - Ensures even utilization of experts
    - Handles cases where too many tokens want the same expert
3. **Parallel Processing**:
    - Tokens are grouped by assigned expert
    - Each expert processes its assigned group
    - Results are combined based on routing weights

The sparse computation pattern makes MoE layers much more efficient than simply running multiple full-size feed-forward layers.

### Intuition Behind MoE

Think of MoE like a team of specialists:

- Instead of every token going through the same general-purpose network
- Tokens are routed to experts that are best suited to process them
- Each expert becomes specialized in handling certain types of patterns
- The router learns to match tokens with the right experts

This specialization allows the model to:

- Handle a wider range of patterns effectively
- Scale capacity without scaling computation for every token
- Develop focused expertise in different aspects of the task

## Source Code

::: directmultistep.model.components.moe
    handler: python
    options:
      show_root_heading: true
      show_source: true
