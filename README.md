# Cluster Optimization 

Course project of Computer Network, where we aim to optimize the task distribution in clusters and the cluster management with Deep Reinforcement Learning (DRL). There are tasks that last for different time and require varied resources in the clusters of big companies such as Google, Amazon, Alibaba, etc. Better allocation of these tasks brings higher service quality and reduces power consumption. Therefore, our group implemented a hierarchical task scheduler for clusters based on the framework of [A Hierarchical Framework of Cloud Resource Allocation and Power Management Using Deep Reinforcement Learning](https://arxiv.org/abs/1703.04221). We recommend you to read the paper before further reading this project.


## Usage
```
python main.py \
-n_servers {number of servers, default is 10} \
-n_tasks {number of tasks, default using all tasks in the dataset}
```
You can refer to `common/argparser.py` to use more available parameters.


## Architecture

The hierarchical framework is as in figure below. It comprises a global tier for VM resource allocation to the servers and a local tier for distributed power management of local servers. And the goal is to achieve the best trade-off between tasks' latency and power/energy consumption in a server cluster. We implemented this complex framework, then analyzed some issues of it and made modification to address them.

<img src="https://github.com/gohsyi/cluster_optimization/blob/master/figures/architecture.png" width="500" height="300" />


### Global Tier

The global tier is in charge of resource allocation. Our framework adopts a continuous-time and event-driven decision framework in which each decision epoch coincides with the arrival time of a new VM (job) request. In this way the action at each decision epoch is simply the target server for VM allocation, which ensures that the available actions are enumerable at each epoch. The continuous-time A2C is chosen as the underlying RL technique in the DRL framework.


### Local Tier

There are two parts in the local tier. The first part is a LSTM model which is to predict the interval time of the jobs. The second part is a RL model to decide the sleeping time based on the number of uncompleted jobs and the prediction in the first part. How it makes the decision is illustrated as below.

<img src="https://github.com/gohsyi/cluster_optimization/blob/master/figures/local.png" width="350" height="300" />


## Discussion

in practice, we find that use a rule-based global tier is more efficient than a RL-based global tier. The rule-based model is described as following: when a new task comes, we dispatch it to the server with the lowest CPU usage. Now we prove that this surrogate method can improve the performance by giving the following reasons.

1. When the global tier adopts Reinforcement Learning, there are at least three deep learning models in the framework: global RL model, local RL model and local workload predictor. Note that they are not independent. The predictor's results are part of the local model's input. The local model's decision (sleep or not) will influence the global model's rewards. While the global model's decision (which machine will the current task be dispatched to) affects the inputs of the predictor. These three models tangle together and it's extremely hard to train one model well while some other model performs poorly. So it's best to let them three converge near the same time, but it's hardly going to happen in practice.

2. Note that the two RL models, global RL model and local RL model, cannot guarantee convergence although there is convergence guarantee of the Q-learning methods. This is because the conditions of Q-learning's convergence includes that the environment has to be stationary, which means the environment will give the same reward and next state when the agent makes the same action at the same state. But due to the inter-influence between the three deep learning models, this stationary condition cannot be guaranteed, and so the convergence of the two RL models.

3. Last but not least, when we use the RL-based global model, most dimensions of the observation is about the state of machines and have nothing to do with the task itself. But global tier needs to dispatch the task to a specific machine conditional on this observation vector. Think about tasks coming continuously, the machine hardly change the state, the input vectors will not differ much. The actions made by the global tier has a great chance to be the same in a time period. So it's common that tasks are all dispatched to only one machines, causing huge latency.

Thus, we propose to use a rule-based model in replace of the global RL model to solve the above address. So that there will only be one RL model in the whole architecture.


## Experiment

We did experiments on dataset `alibaba_clusterdata_v2018` and compared the performance of stochastic model, round robin model, greedy (rule-based) model, hierarchical model and local model (hierarchical model with rule-based global tier).

1. The stochastic model dispatches jobs randomly and the server will not sleep. 
2. The round robin model dispatches jobs in a round-robin way (in turns) and the server will not sleep.
3. The greedy model (rule-based model) dispatches job to the current server with the lowest CPU usage and the server will not sleep.
4. The hierarchical model is the original model we implemented and has been introduced.
5. The local model is the hierarchical model but with global tier replaced by greedy model.

The job latency and power usage are shown below. 

<img src="https://github.com/gohsyi/cluster_optimization/blob/master/figures/latency.png" height="320" width="420" /><img src="https://github.com/gohsyi/cluster_optimization/blob/master/figures/power.png" height="320" width="420" />

The graph on the left is the accumulated job latencycurve. The x-axis represents the number of jobs and y-axis represents latency (units: sec). The graphon the right represents the total power usage of the clusters.  The x-axis represents the number of jobs and y-axis represents power usage (units: kW·h).

The local model can save power almost as much as the hierarchical model but the latency is must less than it.  The hierarchical model that our reference paper proposed has a very large latency and is not very realistic in practice. So our modified model (means local model) achieves best trade-off between job latency and power usage.

To show that our reinforcement learning model is actually learning something. We compare the performances of different training epoches as following.

<img src="https://github.com/gohsyi/cluster_optimization/blob/master/figures/learning.png" height="360" width="460" />
