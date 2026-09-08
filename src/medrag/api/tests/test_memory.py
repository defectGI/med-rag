from medrag.api.memory import ConversationMemory, Message


def test_get_on_unknown_session_returns_empty_list():
    memory = ConversationMemory()
    assert memory.get("unknown") == []


def test_append_and_get_round_trip():
    memory = ConversationMemory()
    memory.append("s1", "user", "merhaba")
    memory.append("s1", "assistant", "merhaba, nasıl yardımcı olabilirim?")
    assert memory.get("s1") == [
        Message(role="user", content="merhaba"),
        Message(role="assistant", content="merhaba, nasıl yardımcı olabilirim?"),
    ]


def test_sessions_are_isolated():
    memory = ConversationMemory()
    memory.append("s1", "user", "a")
    memory.append("s2", "user", "b")
    assert memory.get("s1") == [Message(role="user", content="a")]
    assert memory.get("s2") == [Message(role="user", content="b")]


def test_get_returns_a_copy_not_the_internal_list():
    memory = ConversationMemory()
    memory.append("s1", "user", "a")
    snapshot = memory.get("s1")
    snapshot.append(Message(role="user", content="b"))
    assert memory.get("s1") == [Message(role="user", content="a")]


def test_max_turns_trims_oldest_messages():
    memory = ConversationMemory(max_turns=2)
    for i in range(5):
        memory.append("s1", "user", f"soru {i}")
        memory.append("s1", "assistant", f"cevap {i}")
    # only the last 2 turns (4 messages) remain
    messages = memory.get("s1")
    assert len(messages) == 4
    assert messages[0].content == "soru 3"
    assert messages[-1].content == "cevap 4"


def test_clear_removes_session():
    memory = ConversationMemory()
    memory.append("s1", "user", "a")
    memory.clear("s1")
    assert memory.get("s1") == []


def test_clear_unknown_session_is_a_no_op():
    memory = ConversationMemory()
    memory.clear("unknown")  # must not raise
