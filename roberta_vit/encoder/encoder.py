from transformers import ViTModel, ViTImageProcessor
from transformers import AutoTokenizer, RobertaModel



class VisualEncoder(nn.Module):
    def __init__(self, visual_encoder):
        super(VisualEncoder, self).__init__()
        self.visual_encoder=  visual_encoder
        self.model = ViTModel.from_pretrained(visual_encoder)
        
    def forward(self, inputs):
        outputs = self.model(**inputs)

        return outputs.pooler_output
    

class TextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super(TextEncoder, self).__init__()
        self.text_encoder=  text_encoder
        self.model = RobertaModel.from_pretrained(text_encoder)

    def forward(self, inputs):
        outputs = self.model(**inputs)

        return outputs.pooler_output