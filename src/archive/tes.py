from autolabel import LabelingAgent, AutolabelDataset

agent = LabelingAgent(config='config.json')
ds = AutolabelDataset('comments_from_scraping_new.csv', config = config)
agent.plan(ds)