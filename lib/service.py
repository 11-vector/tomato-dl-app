import typing as tp
from datetime import datetime

from sqlalchemy import create_engine, func

from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
)

from sqlalchemy.types import (
    String
)

import argparse


class Base(DeclarativeBase):
    pass


class ChatHistory(Base):
    __tablename__ = "chat_history"
    id: Mapped[int] = mapped_column(primary_key=True)
    image: Mapped[tp.Optional[str]] = String(length=250)
    message: Mapped[str]
    is_user: Mapped[bool]
    created: Mapped[datetime] = mapped_column(
        server_default=func.current_timestamp())


class Service:

    def store_message(self):
        pass

    def image_inference(self):
        pass


def init_engine():
    engine = create_engine("sqlite:///data.db", echo=False)
    return engine


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="DB"
    )
    parser.add_argument('--init', action="store_true",
                        help="Intialise the database")
    param = parser.parse_args()

    if param.init:
        engine = init_engine()
        Base.metadata.create_all(engine)
