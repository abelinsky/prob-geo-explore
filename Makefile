#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = prob-geo-explore
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Установка зависимостей
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	


## Удаление всех скомпилированных файлов
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 prob_geo_explore
	isort --check --diff prob_geo_explore
	black --check prob_geo_explore

## Форматирование кода
.PHONY: format
format:
	isort prob_geo_explore
	black prob_geo_explore





## Создание окружения
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Разбор файлов
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) prob_geo_explore/dataset.py

## Запуск Problog
.PHONY: problog
problog: data
	$(PYTHON_INTERPRETER) prob_geo_explore/run_problog.py

## Ранжирование блоков
.PHONY: rank_blocks_by_problog
rank_blocks_by_problog: problog
	$(PYTHON_INTERPRETER) prob_geo_explore/rank_and_check.py

## Обучение Markov Logic Network
.PHONY: rank_blocks_by_mln
rank_blocks_by_mln: rank_blocks_by_problog
	$(PYTHON_INTERPRETER) prob_geo_explore/train_mln.py

## Вывод plingo
.PHONY: rank_blocks_by_plingo
rank_blocks_by_plingo: rank_blocks_by_problog
	$(PYTHON_INTERPRETER) prob_geo_explore/plingo.py

## Вывод plingo
.PHONY: evaluate_post_T0
evaluate_post_T0: rank_blocks_by_mln rank_blocks_by_plingo
	$(PYTHON_INTERPRETER) prob_geo_explore/evaluate_post_T0.py


## Сравнение Markov Logic Network и Problog
.PHONY: compare_problog_mln
compare_problog_mln_plingo: evaluate_post_T0
	$(PYTHON_INTERPRETER) prob_geo_explore/compare_problog_mln.py

## Стохастическая оптимизация портфеля (max expected EMV)
.PHONY: optimize_portfolio_stochastic
optimize_portfolio_stochastic: compare_problog_mln_plingo
	$(PYTHON_INTERPRETER) prob_geo_explore/optimize_portfolio.py --mode stochastic

## Робастная оптимизация портфеля (max worst-case EMV)
.PHONY: optimize_portfolio_robust
optimize_portfolio_robust: compare_problog_mln_plingo
	$(PYTHON_INTERPRETER) prob_geo_explore/optimize_portfolio.py --mode robust 

## Визуализация портфельной оптимизации
.PHONY: visualize_portfolios
visualize_portfolios: optimize_portfolio_stochastic optimize_portfolio_robust
	$(PYTHON_INTERPRETER) prob_geo_explore/visualize_portfolios.py 

## Запуск всего пайплайна
.PHONY: run_all
run_all: visualize_portfolios


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
