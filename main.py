from utils.misc import make_env, get_args, set_seed
from runner import Runner

if __name__ == '__main__':
    set_seed(78)
    args = get_args()
    env = make_env(args)
    runner = Runner(args, env)
    if args.evaluate:
        ave_return, norm_return = runner.evaluate(render=True)
    else:
        runner.run()

#### --social_adv --social agent
