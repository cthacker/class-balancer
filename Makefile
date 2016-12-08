lintable := balancer example

.PHONY: pylint
pylint: 
	find $(lintable) -name '*.py' | xargs pylint --rcfile ./.pylintrc -d missing-docstring
