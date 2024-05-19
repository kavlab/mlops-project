# Проект подгруппы 1 семестра 2 по предмету "Администрирование MLOps" УрФУ (ИМО 2023):

- Екатерина З.
- Александр М.
- Александр К.
- Сергей П.
- Денис К.
- Владислав Т.

## Цель проекта

Продемонстрировать навыки, полученные в курсе "Администрирование MLOps".

- Проект оркестируется с помощью CI/CD Jenkins, в котором запускаются модульные тесты и тесты на качество данных, а также осуществляется сборка приложения.
- Датасеты и модель версионируются с помощью DVC и синхронизируются с удалённым хранилищем MinIO.
- Итоговое приложение реализуется в виде образа Docker.


## Состав проекта

Проект состоит из следущих компонентов:
- [скрипты для развертывания сервисов и настройки CI/CD](/buildscripts/README.md) (```./buildscripts```);
- скрипты Python для обрабоки данных и обучения модели (```./src```);
- [приложение проекта](/src/README.md) (```./src/ui```).
