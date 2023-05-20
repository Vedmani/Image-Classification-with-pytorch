import wandb


def initialize_wandb_run(args):
    return wandb.init(project="indian-birds",
                      config=vars(args),
                      save_code=True,
                      resume="allow",
                      name=args.run_id)


def log_values(run, step, **kwargs):
    for key, value in kwargs.items():
        run.log({key: value}, step=step)