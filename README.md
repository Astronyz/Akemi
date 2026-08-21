⚠️ Privacidade e consentimento: Akemi pode capturar áudio do sistema e
observar conteúdo visível na tela. Use somente em dispositivos e contextos
nos quais você tenha autorização; pause a captura quando necessário através
da API local. Nunca publique bases de dados, logs, transcrições, screenshots
ou arquivos .env.

Importante: mesmo rodando com um provedor de cérebro 100% local
(Ollama/llama.cpp), o módulo de self-improve pode ainda assim abrir Pull
Requests públicos no GitHub contendo dados sanitizados extraídos dos logs.
"Local" garante que áudio e tela não saem da máquina para fins de inferência,
mas não significa isolamento total em todas as circunstâncias.

# Akemi

Agente de IA autônomo, open-source, pensado para rodar em background no
Windows nativo (sem Docker/WSL): escutar o áudio do sistema via loopback
WASAPI, observar a tela periodicamente e falar proativamente quando algo
relevante acontece. Também deve se comunicar na rede local com outro agente
(Hermes, em Docker/ZimaOS).

## Estado atual (foundation)

Hoje o repositório cobre a **base de áudio**, não o Jarvis completo:

- captura WASAPI loopback na taxa/canais nativos do dispositivo
- downmix para mono e resample para 16 kHz
- frames de 20 ms compatíveis com webrtcvad
- script de debug que loga quando detecta fala

Ainda **não** estão implementados: STT, visão/OCR, TTS, cérebro (LLM),
API de controle, rede com Hermes e self-improve. As dependências desses
módulos ficam em extras opcionais para o CI e a instalação local não
quebrarem compilando `llama-cpp-python` / `piper-tts` sem necessidade.

## Requisitos

- Windows 10/11
- Python 3.11+

## Instalação

```powershell
git clone https://github.com/Astronyz/Akemi.git
cd Akemi
python -m venv venv
venv\Scripts\activate
pip install -e ".[audio,dev]"
copy .env.example .env
```

Pacote mínimo para desenvolvimento sem hardware de áudio (testes):

```powershell
pip install -e ".[dev]"
pytest
```

Dependências futuras (STT, visão, LLM, TTS, etc.):

```powershell
pip install -r requirements-optional.txt
```

## Executar o debug de áudio

```powershell
python -m akemi.debug
```

Equivalente: `python scripts/run_debug.py` ou `akemi-debug`.

O processo abre o loopback dos alto-falantes padrão, converte o sinal
para PCM 16 kHz mono e imprime `Speech detected` quando o VAD dispara.

## Providers de Cérebro (planejado)

| Provider       | Tipo      | Requisitos externos | Comentário                          |
|----------------|-----------|---------------------|-------------------------------------|
| Anthropic      | Cloud     | Internet, API key   | Requer conta na Anthropic           |
| OpenAI         | Cloud     | Internet, API key   | Requer conta na OpenAI              |
| Ollama         | Local     | Ollama em http://localhost:11434 | Modelo local, sem custos |
| llama.cpp      | Local     | Arquivo .gguf local | Extra `llamacpp`; compilação nativa |

## Self-improve (planejado)

O módulo de auto-melhoria será opcional e ativado via configuração. Ele deve:

- Analisar logs de erro no SQLite
- Gerar diffs via LLM (usando o mesmo provider configurado)
- Aplicar patches em uma worktree git isolada
- Rodar os testes completos
- Abrir um Pull Request via PyGithub com descrição sanitizada
- Ter cooldown de 1 por dia por padrão
- Fazer rollback automático se o serviço crashar após o merge

## Contribuindo

Veja [CONTRIBUTING.md](CONTRIBUTING.md) para detalhes.

## Licença

Este projeto está licenciado sob a Licença Apache 2.0 — veja o arquivo [LICENSE](LICENSE).
