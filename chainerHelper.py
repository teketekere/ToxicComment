import chainer
from chainer import Link, Chain, ChainList, Variable
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset


class MyChain(Chain):
    def __init__(self, outputDim):
        self.output = outputDim
        nunits = 200
        super(MyChain, self).__init__(
            l1=L.Linear(None, nunits),
            l2=L.Linear(None, nunits),
            l3=L.Linear(None, nunits),
            l4=L.Linear(None, self.output)
        )

    def __call__(self, x):
        h1 = F.dropout(F.relu(self.l1(x)))
        h2 = F.dropout(F.relu(self.l2(h1)))
        h3 = F.dropout(F.relu(self.l3(h2)))
        o  = self.l4(h3)
        return o


class ChainerHelper(object):
    def __init__(self, output, epoch=150):
        chainerobject = MyChain(outputDim=output)
        self.model = L.Classifier(chainerobject, lossfun=F.mean_squared_error)
        self.model.compute_accuracy = False
        self.epoch = epoch

    def fit(self, x, y, tx, ty):
        # prepare data
        bsize = 50
        train = tuple_dataset.TupleDataset(x, y)
        train_iter  = chainer.iterators.SerialIterator(train, batch_size=bsize)
        test = tuple_dataset.TupleDataset(tx, ty)
        test_iter = chainer.iterators.SerialIterator(test, batch_size=bsize, repeat=False, shuffle=False)

        # model
        optimizer = chainer.optimizers.AdaDelta(rho=0.9)
        optimizer.setup(self.model)

        updater = training.StandardUpdater(train_iter, optimizer, device=-1)
        trainer = training.Trainer(updater, (self.epoch, 'epoch'), out="result")
        trainer.extend(extensions.Evaluator(test_iter, self.model, device=-1))
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss']))
        trainer.extend(extensions.ProgressBar())

        # learn
        trainer.run()

    def save(self, filename):
        # save model
        chainer.serializers.save_npz(filename, self.model)

    def predict(self, x):
        data = Variable(x)
        predictor = self.model.predictor(data)
        return predictor.data

    def score(self, x, y):
        data = Variable(x)
        pred = self.model.predictor(data)
        score = F.r2_score(pred, y)
        return score.data


    def load(self, filename):
        chainer.serializers.save_npz(filename, self.model)
