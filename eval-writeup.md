# Introduction

The following is a deep dive analysis of running [tau2-bench](https://github.com/sameerbhadouria/tau2-bench) evaluations on `grok-4-fast-reasoning` model. This is a fork of the [original repo] to incorporate some custom work.

**Purpose:**

The tau2 benchmark evaluates conversational agents in a dual control environment. Previous prior art benchmarks simulated single-control environments where only the AI agent had tools to interact with the world, while the user just provided passive information. The dual control environment simulation provides the user tool calling abilities. This simulates a real world scenario where the user needs to actively participate in modifying the state of the world. In many scenarios the agent may not have access to the entire user state and allowing user tool calling can fill this void during simulation.

## Standard Analysis

tau2-bench has a concept of domain specific tasks and for this exercise, the evaluation was done on the airline domain. The agent is required to follow the Airline Agent Policy defined in the `data/tau2/domains/airline/policy.md` file. There are 50 evaluations tasks `data/tau2/domains/airline/tasks.json` with the following category breakdown:

- Cancellation Policy Testing: ~18 tasks
- Flight Modification Testing: ~22 tasks
- Booking Scenarios: ~5 tasks
- Compensation/Complaints: ~5 tasks
- Membership & Benefits: ~3 tasks
- Insurance-Related: ~4 tasks
- Complex Multi-Transaction: ~3 tasks

(Note: Some tasks overlap multiple categories, so the sum is approximate)

For each domain the standard evaluation returns 3 metrics:

1. Average rewards across all samples
2. pass^k Mean pass@k across all tasks
3. Average agent cost across all samples

This benchmark was run with 4 trials for better statistical reliability (also benchmark submission guideline) and using the temperature of 0.0 - a standard value across benchmarking frameworks to reduce randomness.
The above command will also save the results in the file: `data/simulations/grok_airline_all.json`

Steps to run:

1. Follow the setup instructions in README.md
2. Copy `.env.example` and rename to `.env` and add xai api key `XAI_API_KEY=<api_key>`
3. Under the hood tau2-bench uses LiteLLM which requires prefixing the model with `xai` [Reference](https://docs.litellm.ai/docs/providers/xai)
4. Run the tau2 benchmarks using:

   ```bash
   tau2 run \
   --domain airline \
   --agent-llm xai/grok-4-fast-reasoning \
   --user-llm xai/grok-4-fast-reasoning \
   --num-trials 4 \
   --save-to grok_airline_all

   ```

5. To view details of the tasks and the results run: `tau2 view`

---

Metrics Summary and top 3 leaders comparison:

| Rank |         Model         | Pass^1 | Pass^2 | Pass^3 | Pass^4 | % Drop in Pass k=1 to 4 | Avg Cost |
| ---- | :-------------------: | :----: | :----: | :----: | :----: | :---------------------: | :------: |
| 1    |   Claude-Sonnet-4.5   | 70.0%  |   -    |   -    |   -    |            -            |    -     |
| 2    |   Claude-3.7-Sonnet   | 64.2%  | 58.9%  | 55.4%  | 52.1%  |          18.8%          |          |
| 3    |         GPT-5         | 62.5%  | 55.3%  | 51.0%  | 48.0%  |          23.2%          |  $0.134  |
| 4    | Grok-4-fast-reasoning |  60%   | 53.2%  | 49.2%  | 46.4%  |          22.7%          | $0.0157  |
| 4    |    Claude Sonnet 4    |  60%   |   -    |   -    |   -    |            -            |    -     |

Observations:

1. Decent success rate at average reward of 0.6 ~approx 30/50 tasks. Based on the current tau2-bench [Leaderboard](https://taubench.com/#leaderboard) `grok-4-fast-reasoning` would be rank #4 tied with Claude Sonnet 4 and ahead of Claude Opus 4, Qwen3-Max, o4-mini and o3 models.
2. Very economical to run, at $0.0157 per conversation and 50 tasks it costs only $0.80 cents to run 1 trial. This is more than 11x cheaper compared to Open AI GPT-5.
3. The Pass@k seems to be declining with increase in k. There is a drop of 22.7% or 13.6pp from 0.6 @ k=1 to 0.464 @ k=4. This is concerning since we set the temperatue to 0. This also appears to be the case for other models in the leaderboard.

## Deep Dive

The following section is a readout of the analysis done in the `result_analysis/grok_airline_result_analysis.ipynb` notebook.

To understand the failure scenarios better, I used Grok to first classify the various evaluation tasks into categories. Grok was given the policy and task details to categorize the task into one of the following categories:

- Booking Creation
- Flight Modification
- Cancellation Policy
- Compensation/Complaints
- Membership & Benefits
- Insurance Issues
- Complex Multi-Transaction

Further if the task is about user tricking the agent I asked it to append " Tricking" to the categories above.
If the task can't be classified into any of the base categories append " Other" to the categories above.

See `result_analysis/airline_agent_eval_task_classifier.py` for more details.

In the 50 tasks used in our evaluation, we ended up with the following 12 categories and task count

| Category                           | Task Count |
| ---------------------------------- | ---------- |
| Flight Modification                | 11         |
| Cancellation Policy Tricking       | 9          |
| Complex Multi-Transaction          | 8          |
| Flight Modification Tricking       | 6          |
| Compensation/Complaints Tricking   | 4          |
| Booking Creation                   | 3          |
| Complex Multi-Transaction Tricking | 3          |
| Cancellation Policy                | 2          |
| Membership & Benefits              | 1          |
| Insurance Issues Tricking          | 1          |
| Compensation/Complaints            | 1          |
| Insurance Issues                   | 1          |

The following table shows the success rate of the task over 4 trials each consisting of 50 tasks:

| Category                           | Success Count | Total Count | Success Ratio (%) |
| ---------------------------------- | :-----------: | :---------: | :---------------: |
| Cancellation Policy                |       8       |      8      |       100.0       |
| Insurance Issues                   |       4       |      4      |       100.0       |
| Insurance Issues Tricking          |       4       |      4      |       100.0       |
| Membership & Benefits              |       4       |      4      |       100.0       |
| Cancellation Policy Tricking       |      32       |     36      |       88.89       |
| Flight Modification                |      29       |     44      |       65.91       |
| Booking Creation                   |       7       |     12      |       58.33       |
| Flight Modification Tricking       |      14       |     24      |       58.33       |
| Compensation/Complaints Tricking   |       9       |     16      |       56.25       |
| Complex Multi-Transaction          |       8       |     32      |       25.00       |
| Compensation/Complaints            |       0       |      4      |       0.00        |
| Complex Multi-Transaction Tricking |       0       |     12      |       0.00        |

**Observations:**

- Grok achieves a perfect score in Cancellation Policy, Insurance Issues, Insurance Issues Tricking and Membership & Benefits categories.
- On the flip side, it does it really struggles with Complex Multi-Transaction and Compensation/Complaints categories getting almost all of them wrong.

## References:

- [tau2-bench paper](https://arxiv.org/pdf/2506.07982)
- [tau2-bench Leaderboard](https://taubench.com/#leaderboard)
