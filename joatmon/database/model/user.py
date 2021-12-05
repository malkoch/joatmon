from datetime import datetime
from uuid import UUID

from joatmon.database.constraint import (
    CustomConstraint,
    LengthConstraint,
    UniqueConstraint
)
from joatmon.database.document import Document
from joatmon.database.field import Field
from joatmon.database.meta import Meta
from joatmon.database.model.enum import Enum
from joatmon.database.utility import (
    create_new_type,
    email_pattern
)


class EUserType(Enum):
    none = 0

    user = 1
    moderator = 2
    admin = 3


class DUserType(Meta):
    __collection__ = 'd_user_type'

    code = Field(int, nullable=False)
    name = Field(str, nullable=False)
    description = Field(str, nullable=False)


class OUser(Meta):  # need password encryption and decryption
    __collection__ = 'o_user'

    name = Field(str, nullable=False)
    surname = Field(str, nullable=False)
    email = Field(str, nullable=False)
    nickname = Field(str, nullable=False)
    password = Field(str, nullable=False)
    type_id = Field(UUID, nullable=False)

    logged_in = Field(bool, nullable=False)
    logged_in_at = Field(datetime)
    logged_out_at = Field(datetime)

    name_length_constraint = LengthConstraint('name', 5, 64)
    surname_length_constraint = LengthConstraint('surname', 5, 64)
    email_length_constraint = LengthConstraint('email', 5, 64)
    nickname_length_constraint = LengthConstraint('nickname', 5, 64)
    password_length_constraint = LengthConstraint('password', 5, 64)
    email_unique_constraint = UniqueConstraint('email')
    nickname_unique_constraint = UniqueConstraint('nickname')
    email_custom_regex_constraint = CustomConstraint('email', lambda x: email_pattern.search(x))


class OUserLoginAttempt(Meta):
    __collection__ = 'o_user_login_attempt'

    user_id = Field(UUID, nullable=False)
    datetime = Field(datetime, nullable=False)
    success = Field(bool, nullable=False)


class OUserLoginHistory(Meta):
    __collection__ = 'o_user_login_history'

    user_id = Field(UUID, nullable=False)
    login_at = Field(datetime, nullable=False)
    logout_at = Field(datetime)


class OUserTypeHistory(Meta):
    __collection__ = 'o_user_type_history'

    user_id = Field(UUID, nullable=False)
    type_id = Field(UUID, nullable=False)
    started_at = Field(datetime, nullable=False)
    ended_at = Field(datetime)


class OUserNameHistory(Meta):
    __collection__ = 'o_user_name_history'

    user_id = Field(UUID, nullable=False)
    name = Field(str, nullable=False)
    started_at = Field(datetime, nullable=False)
    ended_at = Field(datetime)


class OUserSurnameHistory(Meta):
    __collection__ = 'o_user_surname_history'

    user_id = Field(UUID, nullable=False)
    surname = Field(str, nullable=False)
    started_at = Field(datetime, nullable=False)
    ended_at = Field(datetime)


class OUserEmailHistory(Meta):
    __collection__ = 'o_user_email_history'

    user_id = Field(UUID, nullable=False)
    email = Field(str, nullable=False)
    started_at = Field(datetime, nullable=False)
    ended_at = Field(datetime)


class OUserNicknameHistory(Meta):
    __collection__ = 'o_user_nickname_history'

    user_id = Field(UUID, nullable=False)
    nickname = Field(str, nullable=False)
    started_at = Field(datetime, nullable=False)
    ended_at = Field(datetime)


class OUserPasswordHistory(Meta):
    __collection__ = 'o_user_password_history'

    user_id = Field(UUID, nullable=False)
    password = Field(str, nullable=False)
    started_at = Field(datetime, nullable=False)
    ended_at = Field(datetime)


DUserType = create_new_type(DUserType, (Document,))
OUser = create_new_type(OUser, (Document,))
OUserLoginAttempt = create_new_type(OUserLoginAttempt, (Document,))
OUserLoginHistory = create_new_type(OUserLoginHistory, (Document,))
OUserTypeHistory = create_new_type(OUserTypeHistory, (Document,))
OUserNameHistory = create_new_type(OUserNameHistory, (Document,))
OUserSurnameHistory = create_new_type(OUserSurnameHistory, (Document,))
OUserEmailHistory = create_new_type(OUserEmailHistory, (Document,))
OUserNicknameHistory = create_new_type(OUserNicknameHistory, (Document,))
OUserPasswordHistory = create_new_type(OUserPasswordHistory, (Document,))
