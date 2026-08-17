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

Agente de IA autônomo, open-source, que roda em background no Windows nativo (sem Docker/WSL), escuta o áudio do sistema via loopback WASAPI, observa a tela periodicamente, e fala proativamente por voz quando algo relevante acontece — sem que o usuário precise mandar mensagem manual. Se comunica em rede local com outro agente separado (Hermes, que roda em Docker/ZimaOS).

## Instalação

1. Clone o repositório
2. Crie um ambiente virtual: `python -m venv venv`
3. Ative o ambiente virtual: `venv\Scripts\activate`
4. Instale as dependências: `pip install -r requirements.txt`
5. Copie `.env.example` para `.env` e preencha as variáveis de ambiente necessárias
6. Execute o agente: `python scripts/run_debug.py`

## Providers de Cérebro

| Provider       | Tipo      | Requisitos externos | Comentário                          |
|----------------|-----------|---------------------|-------------------------------------|
| Anthropic      | Cloud     | Internet, API key   | Requer conta na Anthropic           |
| OpenAI         | Cloud     | Internet, API key   | Requer conta na OpenAI              |
| Ollama         | Local     | Ollama rodando em http://localhost:11434 | Modelo local, sem custos |
| llama.cpp      | Local     | Arquivo .gguf local | Requer modelo compatível            |

## Self-improve

O módulo de auto-melhoria é opcional e pode ser ativado via configuração. Ele:
- Analisa logs de erro no SQLite
- Gera diffs via LLM (usando o mesmo provider configurado)
- Aplica patches em uma worktree git isolada
- Roda os testes completos
- Abre um Pull Request via PyGithub com descrição sanitizada
- Tem cooldown de 1 por dia por padrão
- Faz rollback automático se o serviço crashar após o merge

## Contribuindo

Veja [CONTRIBUTING.md](CONTRIBUTING.md) para detalhes.

## Licença

Este projeto está licenciado sob a Licença Apache 2.0 - veja o arquivo [LICENSE](LICENSE) para detalhes.