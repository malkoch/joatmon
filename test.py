from joatmon.language.intent.local import LocalIntent

intents = [
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

model = LocalIntent(intents)
model.train_model()
model.save_model('intent')

for intent in intents:
    for pattern in intent['patterns']:
        result = model.request(pattern)
        print(pattern, intent['name'], result)
