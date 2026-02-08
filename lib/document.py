import typing as tp
from dataclasses import dataclass
import os
import pathlib
import pandas as pd
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


@dataclass
class CropDetail:
    idx: int
    stage: str
    timeline: str
    visual_indicator: str
    common_challenges: str
    actions: str
    notes: str


class CropDataLoader(BaseLoader):

    def __init__(self, filepath: os.PathLike):
        self.filepath = filepath

    def load(self) -> list[Document]:
        return list(self.lazy_load())

    def lazy_load(self) -> tp.Iterator[Document]:
        df = pd.read_excel(self.filepath)
        df.dropna(axis=0, inplace=True)
        datas = map(self._load_data, df.iterrows())
        documents = map(self._tranform_data, datas)
        return documents

    def _load_data(self, value: tuple[int, pd.Series]) -> CropDetail:
        """
        Converts the raw series to structured data
        """
        idx, data = value
        return CropDetail(
            idx=idx,
            stage=data.iloc[0],
            timeline=data.iloc[1],
            visual_indicator=data.iloc[2],
            common_challenges=data.iloc[3],
            actions=data.iloc[4],
            notes=data.iloc[5]
        )

    def _tranform_data(self, data: CropDetail) -> Document:
        """
        Transform the dataset to a valid Document for use with langchain
        """
        page_content = f"Growth Stage: {data.stage}\n" \
            f"Timeline and key Event: {data.timeline}\n" \
            f"Visual Indicators: {data.visual_indicator}\n" \
            f"Common Challenges: {data.common_challenges}\n" \
            f"Actions on the Farm: {data.actions}" \
            f"Notes for Model + NLP: {data.notes}"

        metadata = dict(
            stage=data.stage,
            data=data
        )

        return Document(
            page_content=page_content.strip(),
            metadata=metadata
        )


if __name__ == '__main__':
    BASE_DIR = pathlib.Path.cwd()
    filepath = BASE_DIR / "tomato.xlsx"

    loader = CropDataLoader(filepath)
    dataset = loader.load()
    print(dataset)
