FROM jenkins/jenkins:jdk17

USER root

# Устанавливаем Python и pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Сбрасываем пользователя обратно на jenkins
USER jenkins

EXPOSE 8080 50000

# Команда для запуска Jenkins
CMD ["jenkins.sh"]
