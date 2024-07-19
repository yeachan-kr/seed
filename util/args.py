import os
import argparse

## 모델 학습 관련 Argument (run_classifier.py) ##
class ExperimentArgument:
    def __init__(self):

        data = {}
        parser = self.get_args()
        args = parser.parse_args()
 
        data.update(vars(args))
        self.data = data
        self.set_savename()
        self.__dict__ = data

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", choices=["glass_non_glass", "matscholar", "sofc", "sofc_slot"], required=True, type=str)
        parser.add_argument("--fold", default=None, type=int)

        parser.add_argument("--root", type=str, required=True)
        parser.add_argument("--encoder_class",
                            choices=["bert-base-uncased", "m3rg-iitd/matscibert", "allenai/scibert_scivocab_uncased"],
                            default="m3rg-iitd/matscibert", type=str)

        parser.add_argument("--n_epoch", default=10, type=int)
        parser.add_argument("--seed", default=777, type=int)
        parser.add_argument("--per_gpu_train_batch_size", default=16, type=int)
        parser.add_argument("--per_gpu_eval_batch_size", default=128, type=int)
        parser.add_argument("--gradient_accumulation_step", default=1, type=int)
        parser.add_argument("--seq_len", default=512, type=int)
        parser.add_argument("--warmup_ratio", default=0.1, type=float)
        parser.add_argument("--decay_step", default=20000, type=int)
        parser.add_argument("--clip_norm", default=1.0, type=float)

        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--do_train", action="store_true")
        parser.add_argument("--do_eval", action="store_true")

        parser.add_argument("--evaluate_during_training", action="store_true")
        parser.add_argument("--do_test", action="store_true")
        parser.add_argument('--seed_list', nargs='+', type=int)

        parser.add_argument("--lr", default=1e-5, type=float)
        parser.add_argument("--merge_version", action="store_true")
        parser.add_argument("--bert_freeze", action="store_true")
        parser.add_argument("--contrastive", action="store_true")
        parser.add_argument("--other", action="store_true")

        parser.add_argument("--align_type", choices=["simclr", "cosine"], default="simclr")
        parser.add_argument("--temperature", type=float, default=3.5)

        parser.add_argument("--prototype", choices=["average", "cls"], default="average")

        parser.add_argument("--checkpoint_dir", default="checkpoints", type=str)
        parser.add_argument("--test_log_dir", default="results", type=str)
        parser.add_argument("--vocab_size", type=int)

        parser.add_argument("--transfer_type", choices=["random", "average_input"], default="average_input",
                            type=str)

        return parser

    def set_savename(self):
        if self.data['fold'] is not None:
            self.data["savename"] = os.path.join(self.data["checkpoint_dir"], self.data["dataset"], str(self.data['fold']), str(self.data["seed"]),
                                                self.data["encoder_class"])

        else:
            self.data["savename"] = os.path.join(self.data["checkpoint_dir"], self.data["dataset"], str(self.data["seed"]),
                                                self.data["encoder_class"])


        if self.data["merge_version"]:
            self.data["savename"] += "_{0}_{1}_{2}".format(self.data["transfer_type"], "optimized",
                                                            self.data["vocab_size"])
            if self.data["contrastive"]:
                self.data["savename"] += "_{0}".format(self.data["align_type"])

                if self.data["align_type"] == "simclr":
                    self.data["savename"] += "_{0}".format(self.data["temperature"])

        if self.data["contrastive"]:
            self.data["savename"] += "_{0}".format(self.data["prototype"])
            self.data["savename"] += "_contrastive"

        if self.data["bert_freeze"]:
            self.data["savename"] += "_freeze"

        self.data["vocab_path"] = os.path.join(self.data["root"], self.data["dataset"],
                                            self.data["encoder_class"] + "{0}-merged-vocabulary_optimized_frt".format(
                                            self.data["vocab_size"]))

        if not os.path.isdir(self.data["savename"]):
            os.makedirs(self.data["savename"])

        if self.data["do_test"]:
            if self.data['fold'] is not None:
                self.data["model_path"] = os.path.join(self.data["checkpoint_dir"], self.data["dataset"], str(self.data['fold']), "{0}",
                                                   self.data["encoder_class"])

            else:
                self.data["model_path"] = os.path.join(self.data["checkpoint_dir"], self.data["dataset"], "{0}",
                                                   self.data["encoder_class"])


            self.data["test_dir"] = os.path.join(self.data["test_log_dir"], self.data["encoder_class"])

            if self.data["merge_version"]:
                self.data["model_path"] += "_{0}_{1}_{2}".format(self.data["transfer_type"], "optimized",
                                                                    self.data["vocab_size"])
                self.data["test_dir"] += "{0}/{1}/{2}".format(self.data["transfer_type"], "optimized",
                                                                self.data["vocab_size"])

                if self.data["contrastive"]:
                    self.data["model_path"] += "_{0}".format(self.data["align_type"])

                    if self.data["align_type"] == "simclr":
                        self.data["model_path"] += "_{0}".format(self.data["temperature"])
                        self.data["test_dir"] += "_{0}".format(self.data["temperature"])

                    self.data["model_path"] += "_{0}".format(self.data["prototype"])

                    self.data["test_dir"] += "{0}/".format(self.data["align_type"])

                    self.data["test_dir"] += "_{0}/".format(self.data["prototype"])

                self.data["test_file"] = os.path.join(self.data["test_dir"], self.data["dataset"])
                if self.data['fold'] is not None:
                    self.data['test_file'] += f"_{str(self.data['fold'])}"
                
                if self.data["contrastive"]:
                    self.data["model_path"] += "_contrastive"
                    self.data["test_file"] += "_contrastive"

                    if self.data["bert_freeze"]:
                        self.data["model_path"] += "_freeze"
                        self.data["test_file"] += "_freeze"
                else:
                    self.data["test_file"] += "_vocab_expansion"

            else:
                if self.data['fold'] is not None:
                    self.data["test_file"] = os.path.join(self.data["test_dir"],
                                                      "{0}_{1}_baseline".format(self.data["dataset"], self.data['fold']))
                else:
                    self.data["test_file"] = os.path.join(self.data["test_dir"],
                                                      "{0}_baseline".format(self.data["dataset"]))

            if not os.path.isdir(self.data["test_dir"]):
                os.makedirs(self.data["test_dir"])

            self.data["model_path_list"] = [self.data["model_path"].format(s) for s in self.data["seed_list"]]
            print(self.data["model_path_list"])

## Vocabulary 확장 관련 Argument (avocado.py) ##
class CorpusArgument:
    def __init__(self):
        data = {}
        parser = self.get_args()
        args = parser.parse_args()

        data.update(vars(args))
        self.data = data
        self.set_savename()
        self.__dict__ = data

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", choices=["glass_non_glass", "matscholar", "sofc_slot"],
                            required=True,
                            type=str)

        parser.add_argument("--root", type=str, default="data")
        parser.add_argument("--vocab_size", type=int, default=10000)

        parser.add_argument("--encoder_class",
                            choices=["bert-base-uncased", "m3rg-iitd/matscibert", "allenai/scibert_scivocab_uncased"],
                            default="bert-base-uncased", type=str)

        return parser

    def set_savename(self):
        self.data["vocab_path"] = os.path.join(self.data["root"], self.data["dataset"],
                                            self.data["encoder_class"] + "{0}-merged-vocabulary_optimized_frt".format(
                                            self.data["vocab_size"]))


        if not os.path.isdir(self.data["vocab_path"]):
            os.makedirs(self.data["vocab_path"])
