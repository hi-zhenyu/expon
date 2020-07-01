# expon
Experiment tool for deep learning (PyTorch).


# features

1. Git check

   expon could automatically check the git status of current working directory. 

   In each experiment, it first check the git status (raise expception if the working tree is not clean), and save the current git commit id for code restore and experiment reproduce.

2. Experiment save.

   expon save the all the experment information including the defined metrics, config, experiment seed and perhaps metric visualization (like loss-epochs) in one place. The output supports markdown and html form.

# demo

```

from expon import EXP, Params, Metric

# init. The save directory will be './EXP/run/demo/'
exp = EXP(workspace = 'run', exp_name='demo', exp_description='this is a demo')

# experiment params
params = expon.Params()
params.lamb = 1
params.learning_rate = 0.001
params.batch_size = 512

exp.set_params(params)

# experiment metrics
loss = expon.Metric('loss', 'epoch', draw=True)
acc = expon.Metric('acc', draw=True)
exp.add_metric(loss)
exp.add_metric(acc)

# expon will randomly set the random, numpy and torch seed in [0, 999].
exp.set_seed()

# assume 100 epochs
for i in range(0, 100):
    loss.update(1-0.01*i)
    acc.update(0.01*i)

# add addition information
exp.add_info({'final acc': 0.91})

# save the experiment to one place
exp.save(output_format='md', show_metric=False)
```