import argparse
import atexit
import copy
import datetime
import numpy as np
import os
import torch
import tensorboardX
import sys

from functools import partial


# TODO: Remove this hack
if sys.stdin.isatty():
    print("Appending the artisynth-rl path and prob_mbrl path to python path")
    sys.path.append("/home/praneethsv/Pictures/artisynth-rl/src/python")
    sys.path.append("/home/praneethsv/Pictures/prob_mbrl")

import artisynth_envs
from prob_mbrl import utils, models, algorithms, envs
from prob_mbrl.utils.utilities import extend_args


if __name__ == '__main__':
    args = extend_args()
    loaded_from = args.load_from
    if loaded_from is not None:
        args = torch.load(os.path.join(loaded_from, 'args.pth.tar'))

    # initialize environment
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_flush_denormal(True)
    if args.env in envs.__all__:
        env = envs.__dict__[args.env]()
    else:
        import gym

        env = gym.make(args.env, **vars(args))

    env_name = env.spec.id if env.spec is not None else env.__class__.__name__
    # print('environment name is ', env_name)
    output_folder = os.path.expanduser(args.output_folder)

    results_folder = os.path.join(
        output_folder, "deep_pilco_mm", env_name,
        datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S.%f"))
    try:
        os.makedirs(results_folder)
    except OSError:
        pass
    results_filename = os.path.join(results_folder, "experience.pth.tar")
    torch.save(args, os.path.join(results_folder, 'args.pth.tar'))

    D = env.observation_space.shape[0]
    U = env.action_space.shape[0]
    maxU = env.action_space.high
    minU = env.action_space.low

    # initialize reward/cost function
    if (args.learn_reward or not hasattr(env, 'reward_func')
            or env.reward_func is None):
        reward_func = None
        args.learn_reward = True
    else:
        reward_func = env.reward_func

    # intialize to max episode steps if available
    if hasattr(env, 'spec'):
        if hasattr(env.spec, 'max_episode_steps'):
            args.control_H = env.spec.max_episode_steps
            args.stop_when_done = True
    initial_experience = args.control_H * args.n_initial_epi

    # initialize discount factor
    if args.discount_factor is not None:
        if args.discount_factor == 'auto':
            args.discount_factor = (1.0 / args.control_H) ** (2.0 /
                                                              args.control_H)
        else:
            args.discount_factor = float(args.discount_factor)

    # initialize dynamics model
    dynE = 2 * (D + 1) if args.learn_reward else 2 * D

    # print("control_H value is: ", args.control_H)

    if args.dyn_components > 1:
        """
        Mixture of Gaussian Density Network model. The components have diagonal covariance.
        """
        output_density = models.GaussianMixtureDensity(dynE / 2,
                                                       args.dyn_components)
        dynE = (dynE + 1) * args.dyn_components + 1
    else:
        """"
        Rearranges the incoming dimensions to correspond to the parameters
        of a Gaussian distribution, with diagonal covariance.
        """
        output_density = models.DiagGaussianDensity(dynE / 2)

    '''
    Utility function for creating multilayer perceptrons of varying depth.
    '''
    dyn_model = models.mlp(
        D + U,
        dynE,
        args.dyn_shape,
        dropout_layers=[
            models.modules.CDropout(args.dyn_drop_rate * np.ones(hid))
            if args.dyn_drop_rate > 0 else None for hid in args.dyn_shape
        ],
        nonlin=torch.nn.ReLU)
    """
    DynamicsModel wraps around Regressor in models.core.py 
    :returns two tensors: concatenated states and rewards, states_noise, rewards_noise
    
    """
    dyn = models.DynamicsModel(dyn_model,
                               reward_func=reward_func,
                               output_density=output_density).float()

    """
    initialize policy
    :returns: returns an mlp neural network object with weights initialized 
    """
    pol_model = models.mlp(D,
                           2 * U,
                           args.pol_shape,
                           dropout_layers=[
                               models.modules.BDropout(args.pol_drop_rate)
                               if args.pol_drop_rate > 0 else None
                               for hid in args.pol_shape
                           ],
                           biases_initializer=None,
                           nonlin=torch.nn.ReLU,
                           output_nonlin=partial(models.DiagGaussianDensity,
                                                 U))
    """
    Policy network.
    :returns: returns u (actions/policy) by taking the policy model (pol_model) 
    """
    pol = models.Policy(pol_model, maxU, minU).float()
    print('args\n', args)
    print('Dynamics model\n', dyn)
    print('Policy\n', pol)

    """
    initalize experience dataset
    """
    exp = utils.ExperienceDataset()

    if loaded_from is not None:
        utils.load_checkpoint(loaded_from, dyn, pol, exp)

    # initialize dynamics optimizer
    opt1 = torch.optim.Adam(dyn.parameters(), args.dyn_lr)

    # initialize policy optimizer
    opt2 = torch.optim.Adam(pol.parameters(), args.pol_lr)

    if args.use_cuda and torch.cuda.is_available():
        dyn = dyn.cuda()
        pol = pol.cuda()

    writer = tensorboardX.SummaryWriter(
        logdir=os.path.join(results_folder, "logs"))


    # callbacks
    def on_close():
        writer.close()


    atexit.register(on_close)

    # initial experience data collection
    env.seed(args.seed)
    rnd = lambda x, t: env.action_space.sample()  # noqa: E731

    while exp.n_samples() < initial_experience:
        ret = utils.apply_controller(
            env,
            rnd,
            min(args.control_H, initial_experience - exp.n_samples() + 1),
            stop_when_done=args.stop_when_done)
        exp.append_episode(*ret, policy_params=[])
    if initial_experience > 0:
        exp.policy_parameters[-1] = copy.deepcopy(pol.state_dict())
    exp.save(results_filename)



    """
    policy learning loop
    """
    expl_pol = lambda x, t: (  # noqa: E 731
            pol(x) + args.expl_noise * rnd(x, t)).clip(minU, maxU)
    render_fn = (lambda *args, **kwargs: env.render()) if args.render else None
    for ps_it in range(args.ps_iters):
        """
        apply policy
        """
        new_exp = exp.n_samples() + args.control_H
        while exp.n_samples() < new_exp:
            ret = utils.apply_controller(env,
                                         expl_pol,
                                         min(args.control_H,
                                             new_exp - exp.n_samples() + 1),
                                         stop_when_done=args.stop_when_done,
                                         callback=render_fn)
            exp.append_episode(*ret, policy_params=[])
        exp.policy_parameters[-1] = copy.deepcopy(pol.state_dict())
        exp.save(results_filename)
        print("No of samples ", exp.n_samples())

        """
        train dynamics
        :returns: Returns a dataset where the inputs are state_actions and the outputs
        are next steps.
        """
        X, Y = exp.get_dynmodel_dataset(deltas=True,
                                        return_costs=args.learn_reward)
        # print('state_actions and next steps', X, Y)
        dyn.set_dataset(X.to(dyn.X.device, dyn.X.dtype),
                        Y.to(dyn.X.device, dyn.X.dtype))
        """
        train_regressor is a function that fits/maps state_actions (X) and next steps (Y).
        In doing so, it learns the dynamics of Environment
        """
        utils.train_regressor(dyn,
                              args.dyn_opt_iters,
                              args.dyn_batch_size,
                              True,
                              opt1,
                              log_likelihood=dyn.output_density.log_prob,
                              summary_writer=writer,
                              summary_scope='model_learning/episode_%d' %
                                            ps_it)
        torch.save(dyn.state_dict(),
                   os.path.join(results_folder, 'latest_dynamics.pth.tar'))

        """
        sample initial states for policy optimization
        """
        x0 = exp.sample_states(args.pol_batch_size,
                               timestep=0).to(dyn.X.device,
                                              dyn.X.dtype).detach()

        if args.plot_level > 0:
            utils.plot_rollout(x0[:25], dyn, pol, args.pred_H * 2)


        # train policy
        def on_iteration(i, loss, states, actions, rewards, discount):
            writer.add_scalar('mc_pilco/episode_%d/training loss' % ps_it,
                              loss, i)


        print("Policy search iteration %d" % (ps_it + 1))
        algorithms.mc_pilco(x0,
                            dyn,
                            pol,
                            args.pred_H,
                            opt2,
                            exp,
                            args.pol_opt_iters,
                            discount=args.discount_factor,
                            pegasus=True,
                            mm_states=True,
                            mm_rewards=False,
                            mm_groups=args.mm_groups,
                            maximize=True,
                            clip_grad=args.pol_clip,
                            step_idx_to_sample=args.timesteps_to_sample,
                            init_state_noise=1e-1 * x0.std(0),
                            prioritized_replay=args.prioritized_replay,
                            on_iteration=on_iteration,
                            debug=args.debug)
        torch.save(pol.state_dict(),
                   os.path.join(results_folder, 'latest_policy.pth.tar'))
        if args.plot_level > 0:
            utils.plot_rollout(x0[:25], dyn, pol, args.pred_H * 2)
        writer.add_scalar('robot/evaluation_loss',
                          torch.tensor(ret[2]).sum(), ps_it + 1)
