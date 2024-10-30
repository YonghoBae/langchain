// 호출주소: http://localhost:3000/api/agent/pagebot
// 웹페이지 크롤링을 위한 npm i cheerio 설치필수
import type { NextApiRequest, NextApiResponse } from "next";

//프론트엔드로 반환할 메시지 데이터 타입 참조하기
import { IMemberMessage, UserType } from "@/interfaces/message";

//OpenAI LLM 서비스 객체 참조하기
import { ChatOpenAI } from "@langchain/openai";

//PDF 파일 로더 참조: 물리적 파일존재시 사용
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

//Web사이트상에 존재하는 pdf파일 로드 참조: 예시/참고용

//cheerio 웹페이지 크롤링 라이브러리 참조
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";

//텍스트 스플릿터 객체 참조
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

//임베딩처리를 위한 OpenAI Embedding 객체 참조
//임베딩이란 문장내 단어를 벡터 수치화하는 과정
import { OpenAIEmbeddings } from "@langchain/openai";

//수치화된 벡터 데이터를 저장할 메모리형 벡터저장소 객체 참조
import { MemoryVectorStore } from "langchain/vectorstores/memory";

//시스템,휴먼 메시지 객체를 참조합니다.
import { SystemMessage, HumanMessage } from "@langchain/core/messages";

//프롬프트 템플릿 참조하기
import { ChatPromptTemplate } from "@langchain/core/prompts";

//LLM 응답메시지 타입을 원하는 타입결과물로 파싱(변환)해주는 아웃풋파서참조하기
//StringOutputParser는 AIMessage타입에서 content속성값만 문자열로 반환해주는 파서입니다.
import { StringOutputParser } from "@langchain/core/output_parsers";

//Rag체인, LLM 생성을 위한 모듈 참조
//LangChain Hub는 일종의 오픈소스 저장소처럼 langchain에 특화된 공유된 각종 (RAG전용)프롬프트템플릿 제공
//각종 RAG전용 프롬프트 템플릿들이 제공되며 HUB와 통신하기 위해 pull객체를 참조
import {pull} from "langchain/hub";

//LLM모델에 RAG기반 체인생성 클래스 참조
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";

//서버에서 웹브라우저로 반환하는 처리결과 데이터 타입
type ResponseData = {
  code: number;
  data: string | null | IMemberMessage;
  msg: string;
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<ResponseData>
) {
  //API 호출 기본 결과값 설정
  let apiResult: ResponseData = {
    code: 400,
    data: null,
    msg: "Failed",
  };

  try {
    //클라이언트에서 POST방식 요청해오는 경우 처리
    if (req.method == "POST") {
      //Step1:프론트엔드에서 사용자 프롬프트 추출하기
      const message = req.body.message; //사용자 입력 메시지 추출
      const nickName = req.body.nickName; //사용자 대화명 추출

      //Step2:LLM 모델 생성하기
      const llm = new ChatOpenAI({
        model: "gpt-4o",
        temperature: 0.2,
        apiKey: process.env.OPENAI_API_KEY,
      });

      //step3: cheerio를 이용해 특정 웹페이지 내용을 크롤링실시
      // const loader = new CheerioWebBaseLoader(
      //   "https://www.material-tailwind.com/docs/html/alert"
      // );

      //step3: PDF파일 Indexing 과정
      //step3-1: Indexing과정의 document load 과정
      const loader = new PDFLoader("example_data/Manual.pdf", {
        parsedItemSeparator: "",
      });

      //PDF파일내 페이지 하나당 문서 하나가 생성됨(docs내 doc=pdf page1개)
      const docs = await loader.load();

      //웹페이지 내용 로딩

      //step4: 텍스트 스플릿팅 처리
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
      });

      //pdf document를 지정한 splitter로 단어단위 쪼갠(ChunkData) 집합을 생성
      const splitDOcs = await splitter.splitDocuments(docs);
      //Splitting된 단어의 집합문서를 생성
      // const docs = await splitter.splitDocuments(rawDocs);

      // step3-3: Embeding/Embeding 수치데이터 저장 과정: Splitting된 문서내 단어들을 임베딩(벡터화처리)처리해서 메모리벡터저장소에 저장
      // MemoryVectorStore.fromDocuments(임베딩된문서, 사용할 임베딩모델 처리기);
      // 지정한 임베딩모델을 통해 chunk data를 개별 vector를 수치화하고 수치화된 데이터를 지정한 vector 전용 저장소에 저장
      const vectorStore = await MemoryVectorStore.fromDocuments(
        docs,
        new OpenAIEmbeddings()
      );

      //step4: Query를 통해 벡터저장소에서 사용자 질문과 관련된 검색결과 조회하기
      //메모밀 벡터 저장소에서 사용자 질문으로 Query하기
      //vector저장소 기반 검색기 변수 정의
      //검색기 객체를 생성
      const retriever = vectorStore.asRetriever();

      //RAG기반 사용자 메시지를 이용한 벡터저장소 검색하고 결과 반환
      const searchResult = await retriever.invoke(message);


      //step5: RAG 적용 Prompt와 chain 생성
      //createStuffDocumentsChain()는 LLM모델에 RAG기반 검색 결과를 전달가능한 프롬프트 사용 체인 생성
      //RAG 조회결과를 포함한 전용 프롬프트 체인생성

      //
      const ragPrompt = await pull<ChatPromptTemplate>("rlm/rag-prompt");

      const ragChain = await createStuffDocumentsChain({
        llm:llm,
        prompt:ragPrompt,
        outputParser:new StringOutputParser(),
      });

      //LLM chain을 실행하고 실행시 벡터저장소 검색결과를 추가로 전달해서 llm을 실행
      const resultMessage = await ragChain.invoke({
        question:message,
        context:searchResult,

      })

      //프론트엔드로 반환되는 메시지 데이터 생성하기
      const resultMsg: IMemberMessage = {
        user_type: UserType.BOT,
        nick_name: "bot",
        message: resultMessage,
        send_date: new Date(),
      };

      apiResult.code = 200;
      apiResult.data = resultMsg;
      apiResult.msg = "Ok";
    }
  } catch (err) {
    //Step2:API 호출결과 설정
    const resultMsg: IMemberMessage = {
      user_type: UserType.BOT,
      nick_name: "bot",
      message: "이해할 수 없는 메시지입니다.",
      send_date: new Date(),
    };

    apiResult.code = 500;
    apiResult.data = resultMsg;
    apiResult.msg = "Server Error Failed";
  }

  res.json(apiResult);
}
