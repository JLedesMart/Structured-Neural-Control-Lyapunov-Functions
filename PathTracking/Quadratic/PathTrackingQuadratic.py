import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from docplex.mp.model import Model
from auto_LiRPA.operators import BoundSin
from multiprocessing import Pool
import control
import time
import json
import os

# %% taken from nlc_discrete

env_params = {
    "speed": 2.0,
    "length": 1.0,
    "radius": 10.0,
    "dt": 0.05,  # seconds between state updates
    "max_force": 0.84 #40 degree steering angle, tan(40 degree)
}

seed = 15
train_diameter = 3.0
err_origin = min(train_diameter * 0.05, 0.1)
SAMPLE_SIZE = 1024
STATE_SPACE_SIZE = 2
UB_CONST_GLOBAL_D_E = train_diameter + 0.1
UB_CONST_GLOBAL_THETA_E = train_diameter + 0.1
n_cores_to_use = os.cpu_count() - 2 # in my case 14

def BoundCos_fun(LB, UB):
    sin_op = BoundSin(None, None, 0, {})
    lower_slope, lower_bias, upper_slope, upper_bias = sin_op.bound_relax_branch(torch.tensor([LB + 0.5 * torch.pi]), torch.tensor([UB + 0.5 * torch.pi]))
    return lower_slope, lower_slope * (0.5 * torch.pi) + lower_bias, upper_slope, upper_slope * (0.5 * torch.pi) + upper_bias

# %%

class PolicyNet(nn.Module):
    def __init__(self, width=8, hidden_layers=2):
        super(PolicyNet, self).__init__()
        self.hidden_layers = hidden_layers
        self.width = width
        layers = [nn.Linear(STATE_SPACE_SIZE, width, bias=True), nn.ReLU()]
        for _ in range(hidden_layers):
            layers.append(nn.Linear(width, width, bias = True))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, 1, bias = True))
        self.policy = nn.Sequential(*layers)

    def forward(self, x):
        return self.policy(x)
    
def QuadraticLyapunov(x, S):
    S = torch.as_tensor(S, dtype=x.dtype, device=x.device)
    return ((x@S) * x).sum(dim = -1, keepdim = True)
    
def init_worker(S_arg):
    global policy, S
    policy_arch = torch.load('policy_arch.pt', weights_only=True)
    S = S_arg

    policy = PolicyNet(**policy_arch)
    policy.load_state_dict(torch.load('policy_weights.pt', weights_only=True))
    

def certify_cell_worker(args):
    lb, ub, lbd, ubd = args

    result, ce = certify(policy, S, lb, ub, lbd, ubd)
    return result, ce, lb, ub, lbd, ubd

# %% adapted from nlc_discrete

def weighted_bound_policy(layer, prev_upper, prev_lower, margin = 1e-2):
    prev_mu = (prev_upper + prev_lower)/2
    prev_r = (prev_upper - prev_lower)/2
    mu = layer(prev_mu)
    r = F.linear(prev_r, torch.abs(layer.weight))
    
    upper = mu + r + margin
    lower = mu - r - margin
    return upper, lower


def network_bounds_policy(model, upper, lower):
    with torch.no_grad():
        bounds_list = []
        for layer in model:
            if isinstance(layer, nn.Linear):
                upper, lower = weighted_bound_policy(layer, upper, lower)
                bounds_list.append(np.array([upper, lower])) # store preactivation bounds
            elif isinstance(layer, nn.ReLU):
                upper = torch.relu(upper)
                lower = torch.relu(lower)
                
    return bounds_list, upper, lower


def constraints_policy(cplex_model, model, prefix, x0, bounds):
    if hasattr(model, 'policy'):
        layers = [layer for layer in model.policy.children()]
    else:
        layers = [layer for layer in model.MLP.children()]
    inputs = x0
    input_dims = len(inputs)
    linear_idx = 0
    for i, layer in enumerate(layers):
        
        if isinstance(layer, nn.ReLU):
            continue
        elif isinstance(layer, nn.Linear):
            weight = layer.weight.data.cpu().numpy()
            bias = layer.bias.data.cpu().numpy() if layer.bias is not None else None
            
            next_layer = layers[i+1] if i+1 < len(layers) else None
            next_layer_has_relu = isinstance(next_layer, nn.ReLU)
            layer_out_dim = weight.shape[0]
            
            layer_upper, layer_lower = bounds[linear_idx]
            linear_idx += 1
            
            if next_layer_has_relu:
                post_relu = {}
                
                for neuron in range(layer_out_dim):
                    pre_relu = cplex_model.sum(float(weight[neuron, input_dim]) * inputs[input_dim] for input_dim in range(input_dims))
                    if bias is not None:
                        pre_relu += float(bias[neuron])
                    
                    neuron_ub = layer_upper[neuron]
                    neuron_lb = layer_lower[neuron]
                    
                    if neuron_ub <= 0:
                        post_relu[neuron] = 0
                    elif neuron_lb >= 0:
                        z = cplex_model.continuous_var(name=f'{prefix}_lyx_layer{i}_{neuron}', lb=0, ub=neuron_ub)
                        cplex_model.add_constraint(z == pre_relu)
                        post_relu[neuron] = z
                    else:
                        lyx = cplex_model.continuous_var(name=f'{prefix}_lyx_layer{i}_{neuron}', lb = 0, ub = neuron_ub)
                        lys = cplex_model.continuous_var(name=f'{prefix}_lys_layer{i}_{neuron}', lb = 0, ub = -neuron_lb)
                        lyzx = cplex_model.binary_var(name=f'{prefix}_lyzx_layer{i}_{neuron}')
                        
                        cplex_model.add_constraint(pre_relu == lyx - lys)
                        cplex_model.add_constraint(lyx <= neuron_ub * lyzx)
                        cplex_model.add_constraint(lys <= -neuron_lb * (1 - lyzx))
                        post_relu[neuron] = lyx
                    
               
                inputs = post_relu
                input_dims = layer_out_dim
            else:
                #this corresponds to the output
                output_variables = {}
                for o in range(layer_out_dim):
                    u = cplex_model.continuous_var(name = f'{prefix}_out_{o}', lb = layer_lower[o], ub = layer_upper[o])
                    u_expression = cplex_model.sum(float(weight[o, input_dim]) * inputs[input_dim] for input_dim in range(input_dims))
                    if bias is not None:
                        u_expression += float(bias[o])
                    cplex_model.add_constraint(u == u_expression)
                    output_variables[o] = u
                inputs = output_variables
                input_dims = layer_out_dim
                
    return inputs[0]


def certify(policy_model, S, LB_d_e, UB_d_e, LB_theta_e, UB_theta_e):

    speed = env_params['speed']
    length = env_params['length']
    radius = env_params['radius']
    dt = env_params['dt']
    max_force = env_params['max_force']
    satisfy_cond_flag = True
    print("LB_d_e:", round(LB_d_e, 8), "UB_d_e:", round(UB_d_e, 8), "LB_theta_e:", round(LB_theta_e, 8), "UB_theta_e:", round(UB_theta_e, 8))
    
    band =  int(np.round(0.5*(LB_theta_e + UB_theta_e)/np.pi))
    assert max(abs(UB_theta_e - band*np.pi), abs(band*np.pi - LB_theta_e)) <= np.pi/2, "cos(theta_e) crosses 0 in this cell. Ensure no cells can have both positive and negative cos(theta_e)."
    assert UB_d_e < radius, "d_e upper bound must be smaller than the radius"
    
    lyap_cplex_model = Model(name='Lyapunov Verification')
    lower_bound_state = torch.tensor([LB_d_e, LB_theta_e], dtype= torch.float32)
    upper_bound_state = torch.tensor([UB_d_e, UB_theta_e], dtype= torch.float32)
    bounds_policy, u_upper, u_lower = network_bounds_policy(policy_model.policy, upper_bound_state, lower_bound_state)        

    x_0 = {(i): lyap_cplex_model.continuous_var(name='x_0_{}'.format(i), lb = -lyap_cplex_model.infinity) for i in range(STATE_SPACE_SIZE)} #lb = 0 by default. x_0[0] = theta, x_0[1] = theta_dot

    lyap_cplex_model.add_constraint(x_0[0] <= UB_d_e) # add constraints on the input states
    lyap_cplex_model.add_constraint(x_0[1] <= UB_theta_e)
    lyap_cplex_model.add_constraint(x_0[0] >= LB_d_e)
    lyap_cplex_model.add_constraint(x_0[1] >= LB_theta_e)


    force = lyap_cplex_model.continuous_var(name='force', lb = -lyap_cplex_model.infinity) # what the force actually is after taking into account limits
    force_mid = constraints_policy(lyap_cplex_model, policy_model, 'policy', x_0, bounds_policy)    
    lyap_cplex_model.add_constraint(force == lyap_cplex_model.min(lyap_cplex_model.max(force_mid, -max_force), max_force)) # clipping
    
    #represent the sin function with two lines
    sin_op = BoundSin(None, None, 0, {})
    lower_slope, lower_bias, upper_slope, upper_bias = sin_op.bound_relax_branch(torch.tensor([LB_theta_e]), torch.tensor([UB_theta_e]))
    lower_slope = lower_slope.item()
    lower_bias = lower_bias.item()
    upper_slope = upper_slope.item()
    upper_bias = upper_bias.item()
    
    d_e_acc = lyap_cplex_model.continuous_var(name='d_e_acc', lb = -lyap_cplex_model.infinity)
    lyap_cplex_model.add_constraint(d_e_acc >= speed * (lower_slope * x_0[1] + lower_bias))
    lyap_cplex_model.add_constraint(d_e_acc <= speed * (upper_slope * x_0[1] + upper_bias))
    
    #constrain the cos function linearly
    lower_slope_func, lower_bias_func, upper_slope_func, upper_bias_func = BoundCos_fun(torch.FloatTensor([LB_theta_e]),torch.FloatTensor([UB_theta_e]))
    lower_slope_func = lower_slope_func.item()
    lower_bias_func = lower_bias_func.item()
    upper_slope_func = upper_slope_func.item()
    upper_bias_func = upper_bias_func.item()
    
    theta_e_acc = lyap_cplex_model.continuous_var(name='theta_e_acc', lb = -lyap_cplex_model.infinity)
    
    if band % 2 == 0: # if it is even, cos(theta_e) is >= 0 in this cell
        lyap_cplex_model.add_constraint(theta_e_acc <= speed * force / length - speed * (lower_slope_func * x_0[1] + lower_bias_func) / (radius - LB_d_e))
        lyap_cplex_model.add_constraint(theta_e_acc >= speed * force / length - speed * (upper_slope_func * x_0[1] + upper_bias_func) / (radius - UB_d_e))
        
    else:
       lyap_cplex_model.add_constraint(theta_e_acc <= speed * force / length - speed * (lower_slope_func * x_0[1] + lower_bias_func) / (radius - UB_d_e))
       lyap_cplex_model.add_constraint(theta_e_acc >= speed * force / length - speed * (upper_slope_func * x_0[1] + upper_bias_func) / (radius - LB_d_e))
    
    x_0_next = {(i): lyap_cplex_model.continuous_var(name = 'x_0_next_{}'.format(i), lb = -lyap_cplex_model.infinity) for i in range(STATE_SPACE_SIZE)} #lb = 0 by default
    lyap_cplex_model.add_constraint(x_0_next[0] == x_0[0] + d_e_acc * dt) # constraints on the next state
    lyap_cplex_model.add_constraint(x_0_next[1] == x_0[1] + theta_e_acc * dt)
    

    lyap_cplex_model.parameters.optimalitytarget = 3 # Find the Global optimum
    lyap_cplex_model.parameters.threads.set(1)
    target_x_0 = float(S[0,0])*x_0[0]**2 + float(S[0,1])*x_0[1]*x_0[0] + float(S[1,0])*x_0[0]*x_0[1] + float(S[1,1])*x_0[1]**2
    target_x_0_next = float(S[0,0])*x_0_next[0]**2 + float(S[0,1])*x_0_next[1]*x_0_next[0] + float(S[1,0])*x_0_next[0]*x_0_next[1] + float(S[1,1])*x_0_next[1]**2
    lyap_cplex_model.maximize(target_x_0_next - target_x_0)
    lyap_cplex_model.solve()
    solution_ = lyap_cplex_model._get_solution()

    if solution_ is None:
        print(f"CPLEX FAILED to find a feasible solution for bounds {LB_d_e},{LB_theta_e} to {UB_d_e},{UB_theta_e}.")
        return False, [0.0, 0.0]
    sol_print = round(solution_.objective_value, 8)

    counter_example = solution_.get_values([x_0[0], x_0[1]])
    if sol_print > -0.00001:
        satisfy_cond_flag = False 
        print(f'The decrease condition is satisfied: {satisfy_cond_flag}')
        print(f'The decrease condition is not satisfied at: {counter_example}')
        print(f'The maximum increase found is:{sol_print}')
    else:
        print(f'The decrease condition is satisfied: {satisfy_cond_flag}')
    
    return satisfy_cond_flag, counter_example

def certify_list_all(policy_model, S, certify_counter_example, n_workers = n_cores_to_use):
    torch.save(policy_model.state_dict(), 'policy_weights.pt')
    torch.save({'width': policy_model.width, 'hidden_layers': policy_model.hidden_layers}, 'policy_arch.pt')

    verify_list1 = list(range(-int(train_diameter), -1)) + [-np.pi/2, -1, -0.5, -err_origin]
    verify_list2 = [err_origin, 0.5, 1, np.pi/2] + list(range(2, int(train_diameter + 1)))
    
    cells = []
    
    for idx in range(len(verify_list1) - 1):
        cells.append((-train_diameter, 0, verify_list1[idx], verify_list1[idx+1]))
        cells.append((0, train_diameter, verify_list1[idx], verify_list1[idx+1]))
    for idx in range(len(verify_list2) - 1):
        cells.append((-train_diameter, 0, verify_list2[idx], verify_list2[idx+1]))
        cells.append((0, train_diameter, verify_list2[idx], verify_list2[idx+1]))
    
    cells.append((err_origin, train_diameter/2, -err_origin, err_origin))
    cells.append((train_diameter/2, train_diameter, -err_origin, err_origin))
    cells.append((-train_diameter/2, -err_origin, -err_origin, err_origin))
    cells.append((-train_diameter, -train_diameter/2, -err_origin, err_origin))

    not_certified_list = []
    
    print(f"Starting parallel certification with {n_workers} workers...")
    
    with Pool(processes=n_workers, initializer= init_worker, initargs=(np.array(S),)) as pool:
        results = pool.map(certify_cell_worker, cells)

    not_certified_list = []
    for result, ce, lb, ub, lbd, ubd in results:
        if not result:
            certify_counter_example.append(torch.FloatTensor([ce]))
            not_certified_list.append([lb, ub, lbd, ubd])
    return certify_counter_example, not_certified_list

# %% taken from nlc_discrete

def dynamics(state, action):
    speed = env_params['speed']
    length = env_params['length']
    radius = env_params['radius']
    dt = env_params['dt']
    max_force = env_params['max_force']

    force = torch.clamp(action, min = -max_force, max = max_force)
    d_e = state[:, 0:1]
    theta_e = state[:, 1:2]
    sintheta_e = torch.sin(theta_e)
    costheta_e = torch.cos(theta_e)

    d_e_acc = speed * sintheta_e
    theta_e_acc = (speed * force / length) - (speed * costheta_e / (radius - d_e))
    d_e_next = d_e + dt * d_e_acc
    theta_e_next = theta_e + dt * theta_e_acc
    return torch.cat((d_e_next, theta_e_next), dim = 1)


def lyap_diff(policy_model, S, state):
    
    lyap_value = QuadraticLyapunov(state, S)
    action = policy_model(state)
    state_next = dynamics(state, action)
    lyap_value_next = QuadraticLyapunov(state_next, S)
    lyap_value_diff = lyap_value_next - lyap_value
    reg = torch.sqrt(state[:,0:1]**2 + state[:,1:2]**2)
    return lyap_value_diff, torch.clamp(reg, max = 0.0005)

def gradient_lyap_diff(policy_model, S, state): #to find counterexample of energy increase
    lyap_diff_value, _ = lyap_diff(policy_model, S, state)
    target_func = torch.sum(lyap_diff_value)
    policy_model.zero_grad()
    target_func.backward()
    grad = state.grad
    return torch.sign(grad)


def FindCounterExamples(policy_model, lyap_model, device, steps):
    delta = torch.zeros(SAMPLE_SIZE, STATE_SPACE_SIZE).uniform_(-1, 1)
    min_state = delta * torch.FloatTensor([UB_CONST_GLOBAL_D_E, UB_CONST_GLOBAL_THETA_E])
    min_state = min_state.to(device)
    #steps = 30
    relative_step_size = 1/steps
    for _ in range(steps):
        min_state.requires_grad = True
        min_state = min_state + relative_step_size * gradient_lyap_diff(policy_model, lyap_model, min_state) 
        min_state = min_state.detach()
        min_state[:, 0:1] = torch.clamp(min_state[:, 0:1], min = -UB_CONST_GLOBAL_D_E, max = UB_CONST_GLOBAL_D_E)
        min_state[:, 1:2] = torch.clamp(min_state[:, 1:2], min = -UB_CONST_GLOBAL_THETA_E, max = UB_CONST_GLOBAL_THETA_E)

    lyap_value_diff, _ = lyap_diff(policy_model, lyap_model, min_state)
    min_state_return1 = min_state[lyap_value_diff.flatten() >= 0].clone().detach()
    return min_state_return1

# %%

def lqr(Q, R, dt):
    speed = env_params['speed']
    length = env_params['length']
    radius = env_params['radius']
    dt = env_params['dt']
    
    A = np.array([[0, speed], [-speed/(radius**2), 0]])
    B = np.array([[0],[speed/length]])
    
    Ad = np.eye(2) + A * dt
    Bd = B * dt
    
    K, S, E = control.dlqr(Ad, Bd, Q, R)
    
    return Ad, Bd, K, S, E
    

def generate_data(Q, R, dt, num_samples=100000):
    length = env_params['length']
    radius = env_params['radius']
    Ad, Bd, K, S, E = lqr(Q,R,dt)

    d_e = np.random.uniform(-train_diameter, train_diameter, num_samples)
    theta_e = np.random.uniform(-train_diameter, train_diameter, num_samples)
    x_data = np.stack([d_e, theta_e], axis=1)
    u_labels = (length/radius)-(x_data @ K.T)
    return x_data, u_labels


def split_dataset(X, y):
    
    index = np.arange(len(X))
    np.random.shuffle(index)
    
    X=X[index]
    y=y[index]
    
    split = int(len(X)*0.8)
    
    X_train = X[:split]
    X_test = X[split:]
    
    y_train = y[:split]
    y_test = y[split:]
    
    return X_train, y_train, X_test, y_test

def LQR_init(train_input, train_labels, test_input, test_labels, model, loss_fn, epochs=10, lr=0.01, batch_size=1, print_every=1):

    x_test = test_input
    y_test = test_labels

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=100, factor=0.5)

    loss_dict = {"train": [], "test": []}
    N=len(train_input)

    
    for epoch in range(epochs):
        idx=torch.randperm(N)
        X_shuf=train_input[idx]
        y_shuf=train_labels[idx]

        epoch_loss_sum=0
        for start in range(0, N, batch_size):
            end=min(start + batch_size, N)
            X_batch = X_shuf[start:end]
            y_batch = y_shuf[start:end]
            
            optimizer.zero_grad()
            
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            
            epoch_loss_sum += loss.item() * len(X_batch)

            loss.backward()

            optimizer.step()
            

        loss_dict["train"].append(epoch_loss_sum / N)

        with torch.no_grad():
            y_pred = model(x_test)
            loss = loss_fn(y_pred, y_test)
            loss_dict["test"].append(loss.item())
            scheduler.step(loss)

            if (epoch + 1) % print_every == 0:
                print(epoch + 1, f"Train: {loss_dict['train'][-1]}  Test: {loss.item()}")

    return model, loss_dict



def controller_training(model_policy, S, N_train_steps=500, lr=0.01, verify=True):
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=20, factor=0.5)
    permanent_counter_example = []
    is_success_all = False
    N_rounds = 0
    while not is_success_all:
        optimizer = torch.optim.Adam(list(model_policy.parameters()), lr=lr)
        N = 500
        iter_num = 0
        y = torch.zeros(N, 2).uniform_(-1, 1) * torch.FloatTensor([UB_CONST_GLOBAL_D_E, UB_CONST_GLOBAL_THETA_E])
        if len(permanent_counter_example) != 0:
            y = torch.cat([y] + permanent_counter_example, dim=0)
        y_all = y.clone()
    
        while iter_num < N_train_steps:
    
            lyap_value_diff, reg = lyap_diff(model_policy, S, y)
            Lyapunov_risk = ( F.relu(lyap_value_diff + 0.12)).mean()
    
            optimizer.zero_grad()
            Lyapunov_risk.backward()
            optimizer.step()
    
            if iter_num % 10 == 0:
                result = FindCounterExamples(model_policy, S, 'cpu', 30)
                if len(result) != 0:
                    print(f"iter {iter_num} | counterexamples: {len(result)}")
                else:
                    print(f"iter {iter_num} | conditions satisfied")
                
                #scheduler.step(len(result))
    
                indices = np.random.choice(y_all.size(0), size=min(y_all.size(0), 512))
                y = torch.cat((result, y_all[indices]), dim=0)
                y_all = torch.cat((result, y_all), dim=0)
    
            iter_num += 1
    
        if verify:
            certify_counter_example = []
            certify_counter_example, not_certified_list = certify_list_all(model_policy, S, certify_counter_example)
            if len(not_certified_list) == 0:
                print("FORMALLY VERIFIED over entire training state space")
                is_success_all = True
            else:
                print(f"FAILED — {len(not_certified_list)} regions not certified")
                is_success_all = False
                permanent_counter_example.extend(certify_counter_example)
            print("counterexamples added:", len(certify_counter_example))
        N_rounds +=1

    return model_policy, N_rounds

# %% taken from nlc_discrete


## Find ROA ##


def calc_cut_value(S, diameter):
    min_outside = calc_cut_value_min_outside(S, diameter)
    if min_outside is None:
        return None
    return min_outside - 0.000001

# %%

def calc_cut_value_min_outside(S, diameter):
    V_min_left_right = S[0,0]*diameter**2 - (((S[0,1] + S[1,0])**2)*diameter**2)/(4*S[1,1])
    V_min_top_bottom = S[1,1]*diameter**2 - (((S[0,1] + S[1,0])**2)*diameter**2)/(4*S[0,0])
    return min(V_min_top_bottom, V_min_left_right)

def certify_ring(policy_model, S, certify_counter_example, prev_diameter, new_diameter, n_workers = n_cores_to_use):
    torch.save(policy_model.state_dict(), 'policy_weights.pt')
    torch.save({'width': policy_model.width, 'hidden_layers': policy_model.hidden_layers}, 'policy_arch.pt')
    
    cells = []
    
    zero_crossings = [np.pi/2 + k*np.pi for k in range(-5, 5) if abs(np.pi/2 + k*np.pi) < new_diameter]
    
    theta_e_list = sorted(list(range(-int(new_diameter), int(new_diameter) + 1)) + zero_crossings)
    d_e_list = list(range(-int(prev_diameter), int(prev_diameter) + 1))

    for idx in range(len(theta_e_list) - 1):
        lbd, ubd = theta_e_list[idx], theta_e_list[idx + 1]
        cells.append((-new_diameter, -prev_diameter, lbd, ubd))
        cells.append(( prev_diameter,  new_diameter, lbd, ubd))

    theta_e_list_for_d_e_list = [prev_diameter] + [np.pi/2 + k*np.pi for k in range(0, 5) if np.pi/2 + k*np.pi < new_diameter and np.pi/2 + k*np.pi > prev_diameter] + [new_diameter]

    for idx in range(len(d_e_list) - 1):
        lb, ub = d_e_list[idx], d_e_list[idx + 1]
        for ids in range(len(theta_e_list_for_d_e_list) - 1):
            lbd, ubd = theta_e_list_for_d_e_list[ids], theta_e_list_for_d_e_list[ids + 1]
            cells.append((lb, ub, -ubd, -lbd))
            cells.append((lb, ub,  lbd,  ubd))


    with Pool(processes=n_workers, initializer = init_worker, initargs=(np.array(S),)) as pool:
        results = pool.map(certify_cell_worker, cells)

    not_certified_list = []
    for result, ce, lb, ub, lbd, ubd in results:
        if not result:
            certify_counter_example.append(torch.FloatTensor([ce]))
            not_certified_list.append([lb, ub, lbd, ubd])
            
    return certify_counter_example, not_certified_list

def find_largest_certified_roa(policy_model, S, start_diameter=3.0, increment=1, max_diameter=9.0):
    current_diameter = start_diameter + increment
    prev_diameter = start_diameter
    last_certified_diameter = start_diameter
    certify_counter_example = []


    while current_diameter <= max_diameter:
        print(f"\n--- Trying ring {prev_diameter} -> {current_diameter} ---")
        certify_counter_example, not_certified_list = certify_ring(policy_model, S, certify_counter_example, prev_diameter, current_diameter)

        if len(not_certified_list) == 0:
            print(f"Ring CERTIFIED, certified diameter is now {current_diameter}")
            last_certified_diameter = current_diameter
            prev_diameter = current_diameter
            current_diameter += increment
        else:
            print(f"Ring FAILED — {len(not_certified_list)} regions not certified")
            break

    print(f"\nLargest certified diameter: {last_certified_diameter}")
    cut_value = calc_cut_value(S, last_certified_diameter)

    # plot
    points = np.linspace(-last_certified_diameter, last_certified_diameter, 2000)
    x0_points, x1_points = np.meshgrid(points, points)
    V = float(S[0,0])*x0_points**2 + float(S[0,1])*x1_points*x0_points + float(S[1,0])*x0_points*x1_points + float(S[1,1])*x1_points**2

    region_area = np.sum(V <= cut_value) / (2000 * 2000)
    roa_area_expanded = region_area * last_certified_diameter * last_certified_diameter * 4
    print(f"Certified ROA area: {roa_area_expanded:.2f}")

    plt.figure(figsize=(8, 6))
    plt.contourf(x0_points, x1_points, V, levels=20, cmap='viridis')
    plt.colorbar(label='Lyapunov Value V(x)')
    plt.contour(x0_points, x1_points, V, levels=[cut_value], colors='red', linewidths=3)
    plt.gca().add_patch(plt.Rectangle((-last_certified_diameter, -last_certified_diameter), last_certified_diameter * 2, last_certified_diameter * 2, fill=False, edgecolor='white', linestyle='--', label='Certified Box'))
    plt.xlabel("d_e (Lateral offset)")
    plt.ylabel("Theta_e (Angular error)")
    plt.show()

    return last_certified_diameter, cut_value, roa_area_expanded



if __name__ == '__main__':
    start_time_total = time.perf_counter()

    torch.manual_seed(seed)
    np.random.seed(seed)

    dt = env_params['dt']
    Q = np.array([[1, 0], 
              [0, 1]])
    R = np.array([[10]])
    
    _, _, _, S, _ = lqr(Q, R, dt)
    x_train, u_train, x_test, u_test = split_dataset(*generate_data(Q, R, dt, num_samples=10000))

    x_train = torch.tensor(np.vstack(x_train)).float()
    u_train = torch.tensor(np.vstack(u_train)).float()
    x_test  = torch.tensor(np.vstack(x_test)).float()
    u_test  = torch.tensor(np.vstack(u_test)).float()
    
    total_epochs = 1000
    print_every = 100
    batch_size = 256
    lr = 1e-3
    loss_fn = nn.MSELoss()
    model_policy = PolicyNet()

    start_time_policy_init = time.perf_counter()
    model_policy, loss_dict = LQR_init(x_train, u_train, x_test, u_test, model_policy, loss_fn, epochs=total_epochs, lr=lr, batch_size=batch_size, print_every=print_every)
    end_time_policy_init = time.perf_counter()
    
    N_train_steps=500
    lr = 1e-2


    start_time_training = time.perf_counter()
    model_policy, N_rounds = controller_training(model_policy, S, N_train_steps=N_train_steps, lr=lr, verify=True)
    end_time_training = time.perf_counter()
    
    torch.save(model_policy.state_dict(), 'policy_weights.pt')
    torch.save({'width': model_policy.width, 'hidden_layers': model_policy.hidden_layers}, 'policy_arch.pt')

    points = np.linspace(-train_diameter, train_diameter, 2000)
    x0_points, x1_points = np.meshgrid(points, points)
    V = float(S[0,0])*x0_points**2 + float(S[0,1])*x1_points*x0_points + float(S[1,0])*x0_points*x1_points + float(S[1,1])*x1_points**2

    
    cut_value = calc_cut_value(S, train_diameter)
    region_area = np.sum(V <= cut_value)/(2000 * 2000)
    roa_area = region_area * train_diameter * train_diameter * 4
    plt.figure(figsize=(8, 6))  
    contour_fill = plt.contourf(x0_points, x1_points, V, levels=20, cmap='viridis')
    plt.colorbar(contour_fill, label='Lyapunov Value V(x)')    
    plt.contour(x0_points, x1_points, V, levels=[cut_value], colors='red', linewidths=3)
    plt.gca().add_patch(plt.Rectangle((-train_diameter, -train_diameter), train_diameter*2, train_diameter*2, fill=False, edgecolor='white', linestyle='--', label='Certified Box'))
    plt.xlabel("d_e (Lateral offset)")
    plt.ylabel("Theta_e (Angular error)")
    plt.show()

    start_time_expand = time.perf_counter()
    best_diameter, cut_value_expanded, roa_area_expanded = find_largest_certified_roa(model_policy, S, start_diameter=train_diameter, increment=1, max_diameter=9.0) # increment must be integer
    end_time_expand = time.perf_counter()

    end_time_total = time.perf_counter()

    policy_init_time = end_time_policy_init - start_time_policy_init
    training_time = end_time_training - start_time_training
    expand_time = end_time_expand - start_time_expand
    total_time = end_time_total - start_time_total
    
    results = {
    "seed": seed,
    "train_diameter": train_diameter,
    "err_origin": err_origin,
    "env_params": env_params,

    "trainable_policy_params": sum(p.numel() for p in model_policy.parameters() if p.requires_grad),

    "policy_arch": {
        "width": model_policy.width,
        "hidden_layers": model_policy.hidden_layers,
    },

    "train_box_diameter": train_diameter,
    "cut_value": float(cut_value),
    "roa_area": float(roa_area),

    "expanded_box_diameter": float(best_diameter), 
    "expanded_cut_value": float(cut_value_expanded),
    "expanded_roa_area": float(roa_area_expanded),
    
    "policy_init_time": policy_init_time,
    "training_time": training_time,
    "expand_time": expand_time,
    "total_time": total_time,
    
    "N_rounds": N_rounds,
    }
    base_filename = "PathTracking_Quadratic"
    extension = ".json"

    run_number = 1
    while os.path.exists(f"{base_filename}_{run_number}{extension}"):
        run_number += 1
    
    final_filename = f"{base_filename}_{run_number}{extension}"
    with open(final_filename, "w") as f:
        json.dump(results, f, indent=4)
        
    policy_name = f"{base_filename}_{run_number}_policy.pt"
    
    torch.save(model_policy.state_dict(), policy_name)
        
    print(f"Results successfully saved to {final_filename}")

