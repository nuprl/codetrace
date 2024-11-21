from typing import TypeAlias

from typing import Iterable, List, Optional, Sequence

from django.core.exceptions import ValidationError
from django.utils.translation import ugettext as _
from zerver.lib.exceptions import JsonableError
from zerver.lib.request import JsonableError
from zerver.models import (
    Realm,
    UserProfile,
    get_user_including_cross_realm,
)

__typ0 : TypeAlias = "str"
def raw_pm_with_emails(email_str: __typ0, my_email: __typ0) -> List[__typ0]:
    frags = email_str.split(',')
    emails = [s.strip().lower() for s in frags]
    emails = [email for email in emails if email]

    if len(emails) > 1:
        emails = [email for email in emails if email != my_email.lower()]

    return emails

def user_profiles_from_unvalidated_emails(emails: Iterable[__typ0], realm: Realm) -> List[UserProfile]:
    user_profiles = []  # type: List[UserProfile]
    for email in emails:
        try:
            user_profile = get_user_including_cross_realm(email, realm)
        except UserProfile.DoesNotExist:
            raise ValidationError(_("Invalid email '%s'") % (email,))
        user_profiles.append(user_profile)
    return user_profiles

def get_user_profiles(emails: Iterable[__typ0], realm: Realm) -> List[UserProfile]:
    try:
        return user_profiles_from_unvalidated_emails(emails, realm)
    except ValidationError as e:
        assert isinstance(e.messages[0], __typ0)
        raise JsonableError(e.messages[0])

class __typ1:
    # This is really just a holder for vars that tended to be passed
    # around in a non-type-safe way before this class was introduced.
    #
    # It also avoids some nonsense where you have to think about whether
    # topic should be None or '' for a PM, or you have to make an array
    # of one stream.
    #
    # Eventually we can use this to cache Stream and UserProfile objects
    # in memory.
    #
    # This should be treated as an immutable class.
    def __init__(self, __tmp0: __typ0,
                 user_profiles: Optional[Sequence[UserProfile]]=None,
                 stream_name: Optional[__typ0]=None,
                 topic: Optional[__typ0]=None) -> None:
        assert(__tmp0 in ['stream', 'private'])
        self._msg_type = __tmp0
        self._user_profiles = user_profiles
        self._stream_name = stream_name
        self._topic = topic

    def is_stream(self) -> bool:
        return self._msg_type == 'stream'

    def is_private(self) -> bool:
        return self._msg_type == 'private'

    def user_profiles(self) -> List[UserProfile]:
        assert(self.is_private())
        return self._user_profiles  # type: ignore # assertion protects us

    def stream_name(self) -> __typ0:
        assert(self.is_stream())
        assert(self._stream_name is not None)
        return self._stream_name

    def topic(self) -> __typ0:
        assert(self.is_stream())
        assert(self._topic is not None)
        return self._topic

    @staticmethod
    def legacy_build(sender,
                     message_type_name: __typ0,
                     message_to: Sequence[__typ0],
                     topic_name: __typ0,
                     realm: Optional[Realm]=None) -> 'Addressee':

        # For legacy reason message_to used to be either a list of
        # emails or a list of streams.  We haven't fixed all of our
        # callers yet.
        if realm is None:
            realm = sender.realm

        if message_type_name == 'stream':
            if len(message_to) > 1:
                raise JsonableError(_("Cannot send to multiple streams"))

            if message_to:
                stream_name = message_to[0]
            else:
                # This is a hack to deal with the fact that we still support
                # default streams (and the None will be converted later in the
                # callpath).
                if sender.default_sending_stream:
                    # Use the users default stream
                    stream_name = sender.default_sending_stream.name
                else:
                    raise JsonableError(_('Missing stream'))

            return __typ1.for_stream(stream_name, topic_name)
        elif message_type_name == 'private':
            emails = message_to
            return __typ1.for_private(emails, realm)
        else:
            raise JsonableError(_("Invalid message type"))

    @staticmethod
    def for_stream(stream_name: __typ0, topic: __typ0) -> 'Addressee':
        if topic is None:
            raise JsonableError(_("Missing topic"))
        topic = topic.strip()
        if topic == "":
            raise JsonableError(_("Topic can't be empty"))
        return __typ1(
            __tmp0='stream',
            stream_name=stream_name,
            topic=topic,
        )

    @staticmethod
    def for_private(emails: Sequence[__typ0], realm: <FILL>) -> 'Addressee':
        user_profiles = get_user_profiles(emails, realm)
        return __typ1(
            __tmp0='private',
            user_profiles=user_profiles,
        )

    @staticmethod
    def for_user_profile(user_profile) -> 'Addressee':
        user_profiles = [user_profile]
        return __typ1(
            __tmp0='private',
            user_profiles=user_profiles,
        )

def dummy(a: __typ1) -> __typ1:
    return a