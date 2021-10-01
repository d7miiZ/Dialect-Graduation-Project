def get_writer(trainer):
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, TensorBoardCallback):
            return callback.tb_writer
    raise BrokenPipeError("No TensorboardCallback found in trainer")
    
def write_evaluation_statistics(trainer, dataset):
    writer = get_writer(trainer)
    writer.add_scalars("Evaluation Statistics", trainer.evaluate(dataset))

def write_evaluation_figure(trainer, history):
    """Note: history can be obtained by trainer.state.log_history"""
    writer = get_writer(trainer)
    eval_accuracy = history[["eval_accuracy", "eval_macro_recall", "eval_macro_precision"]].dropna()
    writer.add_figure("Evaluation Statistics", eval_accuracy.plot().get_figure())