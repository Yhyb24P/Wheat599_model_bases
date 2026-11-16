"""Persistent worker for the official compiled DNNGP trainer.

Runs inside the pinned DNNGP3 environment (Python 3.9, TF 2.6).  Protocol
splits arrive as arrays; the official KFold path is bypassed by calling
``dnngp.dnngp_model`` directly.  Per-fit capture records the Keras model,
per-epoch validation loss and learning rates; the refit action replays a
frozen learning-rate sequence for an exact epoch count on merged dev rows.

Protocol: one JSON request per stdin line, one JSON reply per stdout line.
Arrays travel through .npz files; predictions return as .npy files.
"""
import json
import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
# Cap TF thread pools before import: six fit workers share 24 cores with
# other tenants; the default pools oversubscribe the machine.
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

# The compiled trainer and TF print to stdout; keep the JSON protocol on the
# original descriptor and route all library output to stderr.
PROTOCOL_OUT = os.fdopen(os.dup(1), 'w')
os.dup2(2, 1)
sys.stdout = sys.stderr

import numpy as np

CHECKOUT, SCRATCH = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(CHECKOUT, 'Scripts'))

import tensorflow as tf
import dnngp

dnngp.prepare()


class EpochRecorder(tf.keras.callbacks.Callback):
    def __init__(self):
        self.val_loss, self.lr = [], []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if 'val_loss' in logs:
            self.val_loss.append(float(logs['val_loss']))
        self.lr.append(float(tf.keras.backend.get_value(self.model.optimizer.lr)))


def capture_fit(arrays, request):
    """One official inner fit with an extra recorder; returns prediction + traces."""
    captured = {}
    recorder = EpochRecorder()
    original_fit = tf.keras.Model.fit

    def spy(self, *args, **kwargs):
        captured['model'] = self
        kwargs['callbacks'] = list(kwargs.get('callbacks', [])) + [recorder]
        return original_fit(self, *args, **kwargs)

    x_fit = arrays['x_fit'][:, :, None]
    y_fit = arrays['y_fit'][:, None, None]
    tf.keras.Model.fit = spy
    try:
        dnngp.dnngp_model(x_fit, arrays['x_val'][:, :, None], y_fit,
                          arrays['y_val'][:, None, None], request['n_features'],
                          epoch=request['epoch'], patience=request['patience'],
                          batch_size=request['batch_size'], lr=request['lr'],
                          dropout1=request['dropout1'], dropout2=request['dropout2'],
                          output=os.path.join(SCRATCH, 'official'),
                          NMearlystopping=request['earlystopping'],
                          phenoname=request['tag'], part=1)
    finally:
        tf.keras.Model.fit = original_fit
    model = captured['model']
    best = int(np.argmin(recorder.val_loss))
    prediction = model.predict(arrays['x_val'][:, :, None]).ravel()
    return {'prediction': prediction.astype(np.float64),
            'best_epoch': best + 1, 'epochs_run': len(recorder.val_loss),
            'lr_sequence': recorder.lr, 'val_loss': recorder.val_loss}


def refit_replay(arrays, request):
    """Fresh official build, then exact-epoch training with the frozen lr list.

    The official model is captured from a one-epoch trainer call so its
    architecture, initializers, regularizers, loss and optimizer settings are
    exactly the official ones; the initial weights and optimizer state are
    restored, then training proceeds epoch by epoch with the frozen learning
    rates and no validation-based callbacks.
    """
    captured = {}
    original_fit = tf.keras.Model.fit

    def spy(self, *args, **kwargs):
        captured['model'] = self
        if 'initial_weights' not in captured:
            captured['initial_weights'] = self.get_weights()
        return original_fit(self, *args, **kwargs)

    x_fit = arrays['x_fit'][:, :, None]
    y_fit = arrays['y_fit'][:, None, None]
    tf.keras.Model.fit = spy
    try:
        dnngp.dnngp_model(x_fit, x_fit, y_fit, y_fit, request['n_features'],
                          epoch=1, patience=request['patience'],
                          batch_size=request['batch_size'], lr=request['lr'],
                          dropout1=request['dropout1'], dropout2=request['dropout2'],
                          output=os.path.join(SCRATCH, 'official'),
                          NMearlystopping=1, phenoname=request['tag'], part=1)
    finally:
        tf.keras.Model.fit = original_fit
    model = captured['model']
    # The capture call ran one epoch: restore the pristine weights and reset
    # every optimizer variable (iteration counter to 0, accumulators to 0,
    # learning rate to the configured value) so the replay starts from the
    # official initialization.
    model.set_weights(captured['initial_weights'])
    for weight in model.optimizer.weights:
        value = np.zeros_like(tf.keras.backend.get_value(weight))
        if 'lr' in weight.name or 'learning_rate' in weight.name:
            value = np.asarray(request['lr'], dtype=value.dtype)
        tf.keras.backend.set_value(weight, value)
    for epoch, lr in enumerate(request['lr_sequence']):
        tf.keras.backend.set_value(model.optimizer.lr, float(lr))
        model.fit(x_fit, y_fit, batch_size=request['batch_size'],
                  epochs=epoch + 1, initial_epoch=epoch, shuffle=True, verbose=0)
    prediction = model.predict(arrays['x_test'][:, :, None]).ravel()
    return {'prediction': prediction.astype(np.float64),
            'epochs_run': len(request['lr_sequence'])}


def main():
    os.makedirs(os.path.join(SCRATCH, 'official'), exist_ok=True)
    PROTOCOL_OUT.write(json.dumps({'status': 'ready', 'tensorflow': tf.__version__}) + '\n')
    PROTOCOL_OUT.flush()
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('action') == 'shutdown':
            return
        try:
            arrays = dict(np.load(request['input']))
            tf.random.set_seed(int(request['seed']))
            np.random.seed(int(request['seed']))
            if request['action'] == 'refit':
                result = refit_replay(arrays, request)
            else:
                result = capture_fit(arrays, request)
            prediction = result.pop('prediction')
            np.save(os.path.join(SCRATCH, request['tag'] + '_pred.npy'), prediction)
            result['status'] = 'ok'
            PROTOCOL_OUT.write(json.dumps(result) + '\n')
            PROTOCOL_OUT.flush()
        except Exception as error:
            PROTOCOL_OUT.write(json.dumps({'status': 'error', 'error': repr(error)}) + '\n')
            PROTOCOL_OUT.flush()


if __name__ == '__main__':
    main()
