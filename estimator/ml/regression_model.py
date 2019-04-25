# #!/usr/bin/env python3
# """
# Module Docstring
# """
#
# import json
#
# import numpy as np
# import tensorflow as tf
# # ML Imports
# import torch
# from keras.layers import Activation, Dense
# from keras.layers.advanced_activations import LeakyReLU
# from keras.models import Model, Sequential
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.python import keras
#
# LEARNING_RATE = 0.001
#
#
# class RegressionModel:
#     def __init__(self):
#         self._model = None
#         self._history = None
#         self.data
#         pass
#
#     @classmethod
#     def load_model(cls, model_path, hist_path):
#         pass
#
#     def build_model(self, train_dataset):
#         model = keras.Sequential(
#             [
#                 layers.Dense(
#                     64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]
#                 ),
#                 layers.Dense(64, activation=tf.nn.relu),
#                 layers.Dense(1),
#             ]
#         )
#
#         rms_optimizer = tf.keras.optimizers.RMSprop(LEARNING_RATE)
#
#         model.compile(
#             loss="mean_squared_error",
#             optimizer=rms_optimizer,
#             metrics=["mean_absolute_error", "mean_squared_error"],
#         )
#         return model
#
#
# def simple_test_of_model(model, test_data, test_labels, verbose=False):
#     example_result = model.predict(test_data).flatten()
#     assert np.isfinite(example_result).all()
#     assert not np.isnan(example_result).all()
#
#     loss, mae, mse = model.evaluate(test_data[:10], test_labels[:10], verbose=0)
#     if verbose:
#         print("Model Mean Abs Error: +/- {:5.2f} ln like".format(mae))
#     return mae
#
#
# def train_model(
#     model, train_data, train_labels, checkpoint_path, EPOCHS=100, verbose=0
# ):
#     # Callbacks
#
#     # progress callback
#     class PrintDot(keras.callbacks.Callback):
#         """
#         single dot for each completed epoch
#         newline every 100 trained points
#         """
#
#         def on_epoch_end(self, epoch, logs):
#             if epoch % 100 == 0:
#                 print("")
#             print(".", end="")
#
#     # Earlt stop callback
#     early_stop = keras.callbacks.EarlyStopping(
#         monitor="val_loss",  # Training will stop when 'val_loss' stops improving
#         verbose=1,
#         patience=30,  # Num Epochs to check for improvement
#     )
#
#     # create checkpoint callback
#     checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#         checkpoint_path,
#         save_weights_only=True,
#         period=50,  # Save weights, every 50-epochs.
#         verbose=verbose,
#     )
#
#     history = model.fit(
#         x=train_data,
#         y=train_labels,
#         batch_size=128,
#         epochs=EPOCHS,
#         validation_split=0.2,
#         verbose=verbose,
#         callbacks=[checkpoint_callback, early_stop, PrintDot()],
#     )
#
#     return model, history
#
#
# def test_model(
#     model, hist_df, test_df, test_labels, fname="model_testing.html", save_plot=False
# ):
#     enable_plotly_in_cell()
#
#     mae = simple_test_of_model(model, test_df, test_labels, verbose=False)
#
#     test_predictions = model.predict(test_df).flatten()
#
#     mae_train, mae_val = (
#         hist_df["mean_absolute_error"],
#         hist_df["val_mean_absolute_error"],
#     )
#     mse_train, mse_val = (
#         hist_df["mean_squared_error"],
#         hist_df["val_mean_squared_error"],
#     )
#
#     trace1 = go.Scatter(
#         x=test_labels,
#         y=test_predictions,
#         mode="markers",
#         showlegend=False,
#         name="Prediction VS Obs",
#     )
#     best_predection = [test_labels.min(), test_labels.max()]
#     traceTrue = go.Scatter(
#         x=best_predection,
#         y=best_predection,
#         line=dict(color=("rgb(0,0,0)"), dash="dash"),
#         showlegend=False,
#         name="Perfect Predition",
#     )
#     trace2 = go.Histogram(
#         x=test_predictions - test_labels, showlegend=False, name="Error Count"
#     )
#
#     fig = plotly_tools.make_subplots(
#         rows=1,
#         cols=2,
#         subplot_titles=("Predictions vs Observed [ln like]", "Error Distribution"),
#     )
#     fig.append_trace(traceTrue, 1, 1)
#     fig.append_trace(trace1, 1, 1)
#     fig.append_trace(trace2, 1, 2)
#
#     fig["layout"]["xaxis1"].update(title="Observations")
#     fig["layout"]["yaxis1"].update(title="Predictions")
#
#     fig["layout"]["xaxis2"].update(title="Predition Error Amount [ln Like]")
#     fig["layout"]["yaxis2"].update(title="Error Count", type="log", autorange=True)
#
#     fig["layout"].update(title="Testing Model")
#     new_annotations = list(fig["layout"]["annotations"])
#     new_annotations.append(
#         dict(
#             x=test_labels.min(),
#             y=-10,
#             text="Model Mean Abs Err:\n +/- {:5.2f} lnLike".format(mae),
#             showarrow=False,
#         )
#     )
#
#     fig["layout"]["annotations"] = tuple(new_annotations)
#
#     if save_plot == True:
#         plot(fig, filename=fname)  # saves HTML
#     elif save_plot == False:
#         iplot(fig, filename=fname)  # displays
#     else:
#         pass
#
#     pass
#
#
# def plot_training_history(
#     hist: pd.DataFrame, fname="model_training.html", save_plot=False
# ):
#     enable_plotly_in_cell()
#
#     epochs = hist.index.values
#     mae_train, mae_val = (hist["mean_absolute_error"], hist["val_mean_absolute_error"])
#     mse_train, mse_val = (hist["mean_squared_error"], hist["val_mean_squared_error"])
#
#     fig = plotly_tools.make_subplots(
#         rows=1, cols=2, subplot_titles=("Mean Abs Error", "Mean Square Error")
#     )
#
#     # Mean Abs Error Plot
#     trace1 = go.Scatter(
#         x=epochs,
#         y=mae_train,
#         line=dict(color=("rgb(205, 12, 24)"), dash="dash"),
#         name="Train Error",
#         showlegend=False,
#     )
#     trace2 = go.Scatter(
#         x=epochs,
#         y=mae_val,
#         line=dict(color=("rgb(22, 96, 167)")),
#         name="Val Error",
#         showlegend=False,
#     )
#     fig.append_trace(trace1, 1, 1)
#     fig.append_trace(trace2, 1, 1)
#     fig["layout"]["xaxis1"].update(title="Epoch")
#     fig["layout"]["yaxis1"].update(
#         title=r"Mean Abs Error [ ln like ]", type="log", autorange=True
#     )
#
#     # Mean Sqr Error Plot
#     trace3 = go.Scatter(
#         x=epochs,
#         y=mse_train,
#         line=dict(color=("rgb(205, 12, 24)"), dash="dash"),
#         name="Train Error",
#     )
#     trace4 = go.Scatter(
#         x=epochs, y=mse_val, line=dict(color=("rgb(22, 96, 167)")), name="Val Error"
#     )
#     fig.append_trace(trace3, 1, 2)
#     fig.append_trace(trace4, 1, 2)
#     fig["layout"]["xaxis2"].update(title="Epoch")
#     fig["layout"]["yaxis2"].update(
#         title=r"Mean Square Error [ (ln like)^2 ]", type="log", autorange=True
#     )
#
#     fig["layout"].update(title="Training Accuracy")
#
#     if save_plot == True:
#         plot(fig, filename=fname)  # saves HTML
#     elif save_plot == False:
#         iplot(fig, filename=fname)  # displays
#     else:
#         pass
#
#     pass
