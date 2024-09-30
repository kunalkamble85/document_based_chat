import oci

compartment_id = "ocid1.compartment.oc1..aaaaaaaacqxr3j2qbhntidg2s32gb67lv7ydpbfkvcazqvnjfsaxj3vdxr7a"
CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file('./config/config', CONFIG_PROFILE)

# Service endpoint
endpoint = "https://inference.generativeai.uk-london-1.oci.oraclecloud.com"
model_endpoint_map = {"meta.llama3.1-70b":"ocid1.generativeaimodel.oc1.uk-london-1.amaaaaaask7dceyarp4fbl4nicr66ibhaqqxg5w77nnzlgmof5hinslboika",
                      "meta.llama3-70b":"ocid1.generativeaimodel.oc1.uk-london-1.amaaaaaask7dceyaplxvoaiprdoltkphy3fg3ml2xxgt3mwrdptolv5fs5rq",
                      "cohore.command-r-plus":"ocid1.generativeaimodel.oc1.uk-london-1.amaaaaaask7dceyakvoc45z4fz5scsxtactirnhh2icdyuwffp7x3bxkq7fa",
                      "cohore.command-r-16k":"ocid1.generativeaimodel.oc1.uk-london-1.amaaaaaask7dceyauryaezgbyqwehvckgv6sxv3mr7z2l2i4xpbtfoxkemfa"}


generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=config, service_endpoint=endpoint, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240))
chat_detail = oci.generative_ai_inference.models.ChatDetails()

def generate_oci_gen_ai_response(model, messages):
    if model.startswith("cohore.command"):
        print(f"Sending request to model Cohore Command model {model}")
        return handle_cohore_model_request(model, messages)
    else:
        print(f"Sending request to model Meta Llama model {model}")
        return handle_llama_model_request(model, messages)


def handle_llama_model_request(model, messages):
    model_id=model_endpoint_map.get(model)
    all_oci_messages = []
    for message in messages:
        role = message["role"].upper()
        content = message["content"]
        oci_content = oci.generative_ai_inference.models.TextContent()
        oci_content.text = content
        oci_message = oci.generative_ai_inference.models.Message()
        oci_message.role = role
        oci_message.content = [oci_content]
        all_oci_messages.append(oci_message)

    chat_request = oci.generative_ai_inference.models.GenericChatRequest()
    chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
    chat_request.messages = all_oci_messages
    chat_request.max_tokens = 4000
    chat_request.temperature = 0.1
    chat_request.frequency_penalty = 0
    chat_request.presence_penalty = 0
    chat_request.top_p = 0.75
    chat_request.top_k = -1

    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=model_id)
    chat_detail.chat_request = chat_request
    chat_detail.compartment_id = compartment_id
    chat_response = generative_ai_inference_client.chat(chat_detail)
    response = chat_response.data.chat_response.choices[0].message.content[0].text
    return response

def handle_cohore_model_request(model, messages):
    model_id=model_endpoint_map.get(model)
    message_history = []
    for message in messages:
        role = message["role"].upper()
        content = message["content"]
        if role == "USER":
            message_history.append({"role":"USER" , "message": content})
        else:
            message_history.append({"role":"CHATBOT" , "message":content})
    user_last_message = message_history[-1]["message"]
    message_history =message_history[:-1]
    
    chat_request = oci.generative_ai_inference.models.CohereChatRequest()
    chat_request.message = user_last_message
    chat_request.max_tokens = 4000
    chat_request.temperature = 0.1
    chat_request.frequency_penalty = 0
    chat_request.top_p = 0.75
    chat_request.top_k = 0
    chat_request.chat_history = message_history

    chat_detail = oci.generative_ai_inference.models.ChatDetails()
    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=model_id)
    chat_detail.chat_request = chat_request
    chat_detail.compartment_id = compartment_id
    chat_response = generative_ai_inference_client.chat(chat_detail)
    response = chat_response.data.chat_response.chat_history[-1].message
    return response
    


# test code
# generate_oci_gen_ai_response("cohore.command-r-plus",[{"role":"user" , "content":"Hi, my name is Kunal Kamble"}, {"role":"assistant" , "content":"hello Kunal Kamble"}, {"role":"user" , "content":"tell me what is my first name and last name?"}])
# handle_cohore_model_request("cohore.command-r-plus",[{"role":"user" , "content":"Hi, my name is Kunal Kamble"}, {"role":"assistant" , "content":"hello Kunal Kamble"}, {"role":"user" , "content":"tell me what is my first name and last name?"}])