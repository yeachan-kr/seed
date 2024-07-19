import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class Logger:
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        for i, img in enumerate(images):
            # Write the image to a string
            self.writer.add_image(tag=f'{tag}/{i}', img_tensor=img, global_step=step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram(tag=tag, values=values, global_step=step, max_bins=bins)


## Dev set에서 모든 epoch 중 최상의 Evaluation 성능 측정 ##
def log_full_eval_test_results_to_file(args, config, results):
    output_eval_file = os.path.join(args.savename, "all_eval_results.txt")

    with open(output_eval_file, "w") as writer:
        writer.write(
            "total batch size=%d\n"
            % (
                    args.per_gpu_train_batch_size
                    * args.gradient_accumulation_step
            )
        )

        writer.write("train num epochs=%d\n" % args.n_epoch)
        writer.write("learning rate=%f\n" % args.lr)

        writer.write("Dataset name=%s\n" % args.dataset)
        writer.write("augmented vocab size=%d\n" % args.extended_vocab_size)
        writer.write("augmented train ratio=%f\n" % args.aug_ratio)
        writer.write("Bert encoder freeze=%s\n" % args.bert_freeze)
        writer.write("prototype=%s\n" % args.prototype)
        writer.write("Model config %s\n" % str(config))

        best_macro_f1 = -9999.99
        best_accuracy = -9999.99

        for e, result in enumerate(results):
            writer.write("Epoch = %s\n" % (str(e)))
            for key, value in result.items():
                writer.write("%s = %s\n" % (key, str(value)))

                if key == 'macro_f1':
                    if best_macro_f1 < value:
                        best_macro_f1 = value

                if key == 'accuracy':
                    if best_accuracy < value:
                        best_accuracy = value

            writer.write(
                "-------------------------------------------------------\n")

        if args.dataset == "glass_non_glass":
            writer.write("best acc : {0}".format(best_accuracy))    
        else:    
            writer.write("best f1 : {0}".format(best_macro_f1))

## 모델의 Dev set과 Test set에서 Evaluation 성능 측정 ##
def log_full_test_results_to_file(args, test, config, results):
    if test:
        output_eval_file = args.test_file + ".test_results.txt"
    else:
        output_eval_file = args.test_file + ".eval_results.txt"
        
    with open(output_eval_file, "w") as writer:
        writer.write(
            "total batch size=%d\n"
            % (
                    args.per_gpu_train_batch_size
                    * args.gradient_accumulation_step
            )
        )

        writer.write("number of seeds=%d\n" % len(args.seed_list))
        writer.write("Dataset name=%s\n" % args.dataset)
        writer.write("learning rate=%f\n" % args.lr)

        writer.write("augmented vocab size=%d\n" % args.extended_vocab_size)
        writer.write("augmented train ratio=%f\n" % args.aug_ratio)
        writer.write("Bert encoder freeze=%s\n" % args.bert_freeze)
        writer.write("prototype=%s\n" % args.prototype)
        writer.write("Model config %s\n" % str(config))

        for s, result in zip(args.seed_list, results):
            writer.write("seed = %s\n : score : %f\n" % (str(s), float(result)))

            writer.write(
                "-------------------------------------------------------\n")

        average_score = np.mean(results)
        average_std = np.std(results)

        writer.write("average score : %f\n average std : %f" % (float(average_score), float(average_std)))
