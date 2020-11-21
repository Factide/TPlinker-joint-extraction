from tplinker.predict import RelationExtractor
from tplinker.config import eval_config

REL2ID_PATH = "/home/ubuntu/repos/data/data4tplinker/data4bert/nyt_star/rel2id.json"
MODEL_PATH = "/home/ubuntu/repos/TPlinker-joint-extraction/tplinker/default_log_dir/4wfIVy6U/model_state_dict_8.pt"

relation_extractor = RelationExtractor(eval_config, MODEL_PATH, REL2ID_PATH)
predictions = relation_extractor(["Joesph Henlow , a former Gazprom employee has filled a complaint against Turner Media . "])
print(predictions)