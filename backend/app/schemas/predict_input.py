from pydantic import BaseModel

class PredictInput(BaseModel):
    sets_per_match: int
    receives_per_match: int
    blocks_per_match: int
    digs_per_match: int
    attacks_per_match: int