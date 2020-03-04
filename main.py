from trainer import Trainer

trainer = Trainer()

df = trainer.get_data()
trainer.train_keras(df)
trainer.bow('Atendimento horrível')