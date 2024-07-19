import torch
import logging

logger = logging.getLogger(__name__)

## embedding을 초기화 하는 함수 ##
def transfer_embedding(transfer_model, d2p, type):
    logger.info("Transfer representation from pretrained model : {}".format(type))

    def average():
        if hasattr(transfer_model, 'encoder'):
            transfer_layer = transfer_model.encoder.bert.embeddings.word_embeddings if type == "average_input" else transfer_model.main_net.cls.predictions.decoder
            embedding_layer = transfer_model.encoder.bert.embeddings.word_embeddings
        else:
            transfer_layer = transfer_model.main_net.embeddings.word_embeddings if type == "average_input" else transfer_model.main_net.cls.predictions.decoder
            embedding_layer = transfer_model.main_net.embeddings.word_embeddings

        for key, values in d2p.items():
            embedding_id = values[0]
            transfer_ids = values[-1]
            try:
                transfer_embeddings = torch.cat([transfer_layer.weight[t_id].data.view(1, -1) for t_id in transfer_ids], dim=0)
                embedding_layer.weight.data[embedding_id] = torch.mean(transfer_embeddings, dim=0)

            except:
                logger.info("random initialize on %s" % (embedding_id))
                pass

    if type == "average_input" or type == "average_output":
        average()

