
import expon


def test():

    exp = expon.EXP()
    params = expon.Params()
    params.lamb = 1
    params.learning_rate = 0.001
    params.batch_size = 512
    print(params)
    exp.set_params(params)

    loss = expon.Metric('loss', 'epoch', draw=True)
    acc = expon.Metric('acc')
    exp.add_metric(loss)
    exp.add_metric(acc)

    exp.set_seed()

    for i in range(0, 100):
        loss.update(1-0.01*i)
        #acc.update(0.01*i)

    exp.add_info({'final acc': 0.91})
    exp.add_info({'use binary': True})
    exp.add_info({'result': 'right'})
    exp.add_info({'test1': 0.1, 'test2': False, 'test3': 'testing'})
    exp.save()

if __name__ == '__main__':
    test()
    