import time
from datetime import datetime

from google import genai
from google.genai import errors
from utils.preprocessing import get_parsed_data, prepare_data_for_training
import pandas as pd
import logging
from google.api_core import exceptions

logger = logging.getLogger("LLM_LOGGER")
logger.setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)
logger.info("started logging")
client = genai.Client()
RETRIES = 5
UNCONTEXTUALIZED_QUESTION = """
Analyze this song section and classify it as either a verse or chorus:

{part}
keep in mind that choruses are less frequent then verses
Classification:
- Reply with "1" if this is a CHORUS
- Reply with "0" if this is a VERSE
- Reply with only the number, nothing else
"""

CONTEXTUALIZED_QUESTION = """
Analyze this song section and classify it as either a verse or chorus:

TARGET SECTION:
{part}

ADDITIONAL CONTEXT (other parts of the same song in random order):
{context}


Use the additional context to help identify patterns and repetition.
keep in mind that choruses are less frequent then verses

Classification:
- Reply with "1" if the TARGET SECTION is a CHORUS
- Reply with "0" if the TARGET SECTION is a VERSE  
- Reply with only the number, nothing else
"""


def ask(question):
    for _ in range(RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",  # Change this
                contents=question
            )

            logger.debug("successfully answered question %s", question)
            return response.text
        except exceptions.InvalidArgument as e:
            logger.warning(f"invalid argument {e}")
        except errors.ClientError as e:
            raise e
        except exceptions.Unauthenticated as e:
            logger.error(f"Unauthenticated (401): {e}")
        except exceptions.PermissionDenied as e:
            logger.error(f"Permission denied (403): {e}")
        except exceptions.NotFound as e:
            logger.error(f"Not found (404): {e}")
        except exceptions.TooManyRequests as e:
            time.sleep(10)
        except exceptions.InternalServerError as e:
            time.sleep(10)
        except Exception as e:
            logger.warning(f"Other error: {e}")
        time.sleep(2)
    raise Exception


def context_resp(text, context):
    try:
        response = ask(CONTEXTUALIZED_QUESTION.format(part=text, context=context))
        return int(response)
    except:
        pass


def no_context_resp(text):
    try:
        response = ask(UNCONTEXTUALIZED_QUESTION.format(part=text))
        return int(response)
    except:
        pass


data = []
parsed_data = get_parsed_data(350, "test")
df = prepare_data_for_training(parsed_data)
print(f"{len(df)} rows overall")
for i, row in enumerate(df.itertuples(index=False)):
    time.sleep(2)
    try:
        row_data = [row.song_id, row.text, row.context, row.label, no_context_resp(row.text),
                    context_resp(row.text, row.context)]
        if row_data[-1] is not None and row_data[-2] is not None:
            data.append(row_data)
        else:
            print(row_data)
    except:
        break
data = pd.DataFrame(data, columns=["song_id", "text", "context", "label", "no_context", "with_context"])
data.to_excel(f"LLM_{str(datetime.now()).replace(':', '').replace('-', '').replace('.', '')}.xlsx", index=False)
