from pydantic import BaseModel


class RadiologyAssessment(BaseModel):
    question: str
    has_enlarged_cardiomediastinum: bool
    has_cardiomegaly:bool
    has_lung_lesion:bool
    has_lung_opacity:bool
    has_edema:bool
    has_consolidation:bool
    has_pneumonia:bool
    has_atelectasis: bool
    has_pneumothorax: bool
    has_pleural_effusion: bool
    has_pleural_other: bool
    has_fracture: bool
    has_support_devices: bool
    description: str
