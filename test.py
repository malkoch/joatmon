from joatmon.nn.processing import (
    OneHotEncoder,
    pad_sequences,
    Tokenizer,
    word_tokenize
)

intents = [
    {
        "name": "",
        "patterns": [
            ""
        ]
    },
    {
        "name": "enable",
        "patterns": [
            "enable"
        ]
    },
    {
        "name": "disable",
        "patterns": [
            "disable"
        ]
    },
    {
        "name": "update",
        "patterns": [
            "update"
        ]
    },
    {
        "name": "delete",
        "patterns": [
            "delete"
        ]
    },
    {
        "name": "configure",
        "patterns": [
            "configure"
        ]
    },
    {
        "name": "start",
        "patterns": [
            "start"
        ]
    },
    {
        "name": "stop",
        "patterns": [
            "stop"
        ]
    },
    {
        "name": "restart",
        "patterns": [
            "restart"
        ]
    },
    {
        "name": "skip",
        "patterns": [
            "skip"
        ]
    },
    {
        "name": "help",
        "patterns": [
            "help"
        ]
    },
    {
        "name": "exit",
        "patterns": [
            "exit"
        ]
    },
    {
        "name": "activate",
        "patterns": [
            "activate",
            "hey eva",
            "eva"
        ]
    }
]

x_values = []
y_values = []

data = []
for intent in intents:
    for pattern in intent.get('patterns', []):
        if intent.get('name', None):
            x_values.append(pattern)
            y_values.append(intent.get('name', None))

encoder = OneHotEncoder()
encoder.fit(y_values)

tokenized = [word_tokenize(d) for d in x_values]

tokenizer = Tokenizer()
tokenizer.fit_on_text(tokenized)

x_values = list(tokenizer.text_to_sequence(tokenized))
y_values = encoder.transform(y_values)

x_values = pad_sequences(x_values)

print(x_values)
print(y_values)
