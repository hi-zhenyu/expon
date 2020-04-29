
from expon.core.exp import EXP
from expon.core.params import Params
from expon.core.metric import Metric

def test():

    exp = EXP()
    params = Params()
    params.lamb = 1
    params.learning_rate = 0.001
    params.batch_size = 512
    print(params)
    exp.set_params(params)

    loss = Metric('loss', 'epoch')
    acc = Metric('loss')
    exp.add_metric(loss)
    exp.add_metric(acc)

    exp.set_seed(123)

    for i in range(0, 100):
        loss.update(1-0.01*i)
        acc.update(0.01*i)

    exp.save(output_format='html')

if __name__ == '__main__':
    test()
    