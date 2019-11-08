import argparse
from prob_mbrl import utils

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def extend_args():
    parser = argparse.ArgumentParser("Deep-PILCO with moment matching")
    parser.add_argument('-e', '--env', type=str, default="Cartpole")
    parser.add_argument('-o',
                        '--output_folder',
                        type=str,
                        default="~/.prob_mbrl/")
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--n_initial_epi', type=int, default=0)
    parser.add_argument('--load_from', type=str, default=None)
    parser.add_argument('--pred_H', type=int, default=15)
    parser.add_argument('--control_H', type=int, default=40)
    parser.add_argument('--discount_factor', type=str, default=None)
    parser.add_argument('--prioritized_replay', action='store_true')
    parser.add_argument('--timesteps_to_sample',
                        type=utils.load_csv,
                        default=0)
    parser.add_argument('--mm_groups', type=int, default=None)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--dyn_lr', type=float, default=1e-4)
    parser.add_argument('--dyn_opt_iters', type=int, default=2000)
    parser.add_argument('--dyn_batch_size', type=int, default=100)
    parser.add_argument('--dyn_drop_rate', type=float, default=0.1)
    parser.add_argument('--dyn_components', type=int, default=1)
    parser.add_argument('--dyn_shape', type=utils.load_csv, default=[200, 200])

    parser.add_argument('--pol_lr', type=float, default=1e-3)
    parser.add_argument('--pol_clip', type=float, default=1.0)
    parser.add_argument('--pol_drop_rate', type=float, default=0.1)
    parser.add_argument('--pol_opt_iters', type=int, default=1000)
    parser.add_argument('--pol_batch_size', type=int, default=100)
    parser.add_argument('--ps_iters', type=int, default=100)
    parser.add_argument('--pol_shape', type=utils.load_csv, default=[200, 200])

    parser.add_argument('--plot_level', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--learn_reward', action='store_true')
    parser.add_argument('--keep_best', action='store_true')
    parser.add_argument('--stop_when_done', action='store_true')
    parser.add_argument('--expl_noise', type=float, default=0.0)

    # ArtiSynth Env Args
    parser.add_argument('--gui', type=str2bool, default=True,
                        help='run environment with GUI.')
    parser.add_argument('--verbose', type=int, default='20',
                        help='Verbosity level')
    parser.add_argument('--ip', type=str, default='localhost',
                        help='IP of server')
    parser.add_argument('--port', type=int, default=8080,
                        help='port to run the server on (default: 4545)')
    parser.add_argument('--incremental_actions', type=str2bool, default=False,
                        help='Treat actions as increment/decrements to the current excitations.')
    parser.add_argument('--reset_step', type=int, default=1e10, help='Reset envs every n iters.')
    parser.add_argument('--include_current_state', type=str2bool, default=True,
                        help='Include the current position/rotation of the model in the state.')
    parser.add_argument('--include_current_excitations', type=str2bool, default=True,
                        help='Include the current excitations of actuators in the state.')
    parser.add_argument('--goal_threshold', type=float, default=0.1,
                        help='Difference between real and target which is considered as success when reaching a goal')
    parser.add_argument('--goal_reward', type=float, default=0, help='The reward to give if goal was reached.')
    parser.add_argument('--zero_excitations_on_reset', type=str2bool, default=True,
                        help='Reset all muscle excitations to zero after each reset.')
    parser.add_argument('--w_u', type=float, default=1, help='weight of distance reward term')
    parser.add_argument('--w_d', type=float, default=1, help='weight of damping reward term')
    parser.add_argument('--w_r', type=float, default=1, help='weight of excitation regularization reward term')
    parser.add_argument('--wait_action', type=float, default=0.0,
                        help='Wait (seconds) for action to take place and environment to stabilize.')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='eval interval, one eval per n updates (default: 100)')
    parser.add_argument('--test', type=str2bool, default=False, help='Evaluate a trained model.')

    # parameters
    args = parser.parse_args()
    return args
