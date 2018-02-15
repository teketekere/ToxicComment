import chainer
from chainer import Link, Chain, ChainList, Variable
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset


class MyChain(Chain):
    def __init__(self, inputDim, outputDim):
        self.input = inputDim
        self.output = outputDim
        super(MyChain, self).__init__(
            l1=L.Linear(self.input, 512),
            l2=L.Linear(512, 256),
            l3=L.Linear(256, 64),
            l4=L.Linear(64, self.output)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        o  = self.l4(h3)
        return o


class ChainerHelper(object):
    def __init__(self, input, output, epoch=10):
        chainerobject = MyChain(inputDim=input, outputDim=output)
        self.model = L.Classifier(chainerobject, lossfun=F.mean_squared_error)
        self.model.compute_accuracy = False
        self.epoch = epoch
        
    def fit(self, x, y, tx, ty):
        # prepare data
        bsize = 100
        train = tuple_dataset.TupleDataset(x, y)
        train_iter  = chainer.iterators.SerialIterator(train, batch_size=bsize)
        test = tuple_dataset.TupleDataset(tx, ty)
        test_iter = chainer.iterators.SerialIterator(test, batch_size=bsize, repeat=False, shuffle=False)

        # model
        optimizer = chainer.optimizers.Adam()
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
