import sys
import threading
import warnings

if sys.platform != 'win32':
    warnings.warn('The microphone module can only be used on a Windows system. STT will never be enabled')


class InputDriver:
    def __init__(self, output_device, stt_enabled, recognizer='SAPI.SpInProcRecognizer', words=[]):
        super(InputDriver, self).__init__()

        self.output_device = output_device
        self.stop_event = threading.Event()
        self.stt_enabled = stt_enabled and sys.platform == 'win32'

        if self.stt_enabled:
            import win32com.client
            from win32com.client import constants

            if recognizer == 'SAPI.SpInProcRecognizer':
                self.listener = win32com.client.Dispatch("SAPI.SpInProcRecognizer")
                self.listener.AudioInputStream = win32com.client.Dispatch("SAPI.SpMMAudioIn")
                self.listener_base = win32com.client.getevents("SAPI.SpInProcRecoContext")
            elif recognizer == 'SAPI.SpSharedRecognizer':
                self.listener = win32com.client.Dispatch("SAPI.SpSharedRecognizer")
                self.listener_base = win32com.client.getevents("SAPI.SpSharedRecoContext")
            else:
                raise ValueError('listener is not recognized')

            class ListenerEvents(self.listener_base):
                OnRecognition = self.on_recognition

            self.context = self.listener.CreateRecoContext()
            self.grammar = self.context.CreateGrammar()

            if words:
                self.grammar.DictationSetState(0)
                self.words = self.grammar.Rules.Add("words", constants.SRATopLevel + constants.SRADynamic, 0)
                self.words.Clear()
                for word in words:
                    self.words.InitialState.AddWordTransition(None, word)
                self.grammar.Rules.Commit()
                self.grammar.CmdSetRuleState("words", 1)  # wordsRule
                self.grammar.Rules.Commit()
            else:
                self.grammar.DictationSetState(1)

            self.listener_event_handler = ListenerEvents(self.context)

    def on_recognition(self, _1, _2, _3, result):
        import win32com.client
        new_result = win32com.client.Dispatch(result)
        print("You said: ", new_result.PhraseInfo.GetText())

    def input(self, prompt=None):
        p = prompt if prompt is not None else 'default prompt: '
        self.output_device.output(p)
        return self.readline()

    def readline(self):
        if not self.stt_enabled:
            if self.output_device.tts_enabled:
                sys.stdout.write('~ Your Command ->')
                sys.stdout.flush()
            return sys.stdin.readline()
        else:
            return ''

    def stop(self):
        self.stop_event.set()
