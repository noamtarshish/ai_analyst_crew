from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from ai_analyst_crew.tools.sentiment_tools import analyze_sentiment_from_news
from ai_analyst_crew.tools.technical_tools import forecast_price, forecast_with_model
from ai_analyst_crew.tools.fundamental_tools import analyze_fundamentals


# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class AiAnalystCrew():
    """AiAnalystCrew crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # AGENTS

    @agent
    def technical_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_analyst'], # type: ignore[index]
            tools=[forecast_with_model],
            verbose=True
        )

    @agent
    def sentiment_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['sentiment_analyst'], # type: ignore[index]
            tools=[analyze_sentiment_from_news],
            verbose=True
        )

    @agent
    def fundamental_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['fundamental_analyst'], # type: ignore[index]
            tools=[analyze_fundamentals],
            verbose=True
        )




    # TASKS

    @task
    def technical_forecast_task(self) -> Task:
        return Task(
            config=self.tasks_config['technical_forecast_task'], # type: ignore[index]
        )

    @task
    def sentiment_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['sentiment_analysis_task'], # type: ignore[index]
        )

    @task
    def fundamental_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['fundamental_analysis_task'], # type: ignore[index]
        )



    @crew
    def crew(self) -> Crew:
        """Creates the AiAnalystCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=[self.technical_analyst(),
                    self.sentiment_analyst(),
                    self. fundamental_analyst()
                    ],
            tasks=[self.technical_forecast_task(),
                   self.sentiment_analysis_task(),
                   self.fundamental_analysis_task()
                   ],
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
