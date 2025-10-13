from data.dataset import PasswordDataset
from utils.preprocessing import Preprocessor
from train.train import Trainer
from utils.guessing import Guesser

data_path = "data/trainset/rockyou_new.txt"
config_path = "config/config.json"

# 创建并运行训练器
trainer = Trainer(config_path=config_path, data_path=data_path)
trainer.train()
print("done")
# guesser = Guesser(model=trainer.model, preprocessor=trainer.preprocessor)
# password = "example_password"
# probability = guesser.guess_one_pwd(password)
# print(f"Probability of password '{password}': {probability:.6e}")
# password = "abcdefghhhhhhh"
# probability = guesser.guess_one_pwd(password)
# print(f"Probability of password '{password}': {probability:.6e}")