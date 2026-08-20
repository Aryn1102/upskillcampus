loader = DataLoader(...)
df = loader.load_data()

preprocessor = DataPreprocessor(df)
df = preprocessor.preprocess()

eda = EDA(df)
eda.target_distribution()
eda.correlation_heatmap()
eda.feature_distributions()
eda.run()

trainer = ModelTrainer(df)
trainer.train()

