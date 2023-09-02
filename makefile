SHELL = /bin/bash
.PHONY: docs

docs:
	pipenv run pdoc calidhayte -d numpy -o docs/ --math --mermaid --search
