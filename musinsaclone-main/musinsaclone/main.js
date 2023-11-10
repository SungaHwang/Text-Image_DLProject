'use strict';

// nav menu list dispaly handler
const navListHandler = {
  mouseover: function() {
    switch(this) {
      case rankLi || rankLi_:
        rankLi_.style.display = "block";
        break;
      case updateLi || updateLi_:
        updateLi_.style.display = "block";
        break;
      case styleLi || styleLi_:
        styleLi_.style.display = "block";
        break;
      case saleLi || saleLi_:
        saleLi_.style.display = "block";
        break;
      case specialLi || specialLi_:
        specialLi_.style.display = "block";
        break;
      case magazineLi || magazineLi_:
        magazineLi_.style.display = "block";
        break;
      case tvLi || tvLi_:
        tvLi_.style.display = "block";
        break;
      case eventLi || eventLi_:
        eventLi_.style.display = "block";
        break;
      default:
        break;
    }
  },
  mouseout: function() {
    switch(this) {
      case rankLi || rankLi_:
        rankLi_.style.display = "none";
        break;
      case updateLi || updateLi_:
        updateLi_.style.display = "none";
        break;
      case styleLi || styleLi_:
        styleLi_.style.display = "none";
        break;  
      case saleLi || saleLi_:
        saleLi_.style.display = "none";
        break;
      case specialLi || specialLi_:
        specialLi_.style.display = "none";
        break;
      case magazineLi || magazineLi_:
        magazineLi_.style.display = "none";
        break;
      case tvLi || tvLi_:
        tvLi_.style.display = "none";
        break;
      case eventLi || eventLi_:
        eventLi_.style.display = "none";
        break;
      default:
        break;
    }
  }
};

// nav 목록 및 hover시 show 제어
// 랭킹 목록 show 제어
const rankLi = document.querySelector(".rankLi");
const rankLi_ = document.querySelector(".rankLi_");
const rankEvent = new AddHoverEvent(rankLi, rankLi_);
// 업데이트 목록 show 제어
const updateLi = document.querySelector(".updateLi");
const updateLi_ = document.querySelector(".updateLi_");
const updateEvent = new AddHoverEvent(updateLi, updateLi_);
// 코디 목록 show 제어
const styleLi = document.querySelector(".styleLi");
const styleLi_ = document.querySelector(".styleLi_");
const styleEvent = new AddHoverEvent(styleLi, styleLi_);
// 세일 목록 show 제어
const saleLi = document.querySelector(".saleLi");
const saleLi_ = document.querySelector(".saleLi_");
const saleEvent = new AddHoverEvent(saleLi, saleLi_);
// 스페셜 목록 제어
const specialLi = document.querySelector(".specialLi");
const specialLi_ = document.querySelector(".specialLi_");
const specialEvent = new AddHoverEvent(specialLi, specialLi_);
// 매거진 목록 제어
const magazineLi = document.querySelector(".magazineLi");
const magazineLi_ = document.querySelector(".magazineLi_");
const magazineEvent = new AddHoverEvent(magazineLi, magazineLi_);
// TV 목록 제어
const tvLi = document.querySelector(".tvLi");
const tvLi_ = document.querySelector(".tvLi_");
const tvEvent = new AddHoverEvent(tvLi, tvLi_);
// 이벤트메뉴 목록 제어
const eventLi = document.querySelector(".eventLi");
const eventLi_ = document.querySelector(".eventLi_");
const eventMenuEvent = new AddHoverEvent(eventLi, eventLi_);

// nav 목록 event추가 생성자함수
function AddHoverEvent(mainList, subList) {
  mainList.addEventListener("mouseover", navListHandler.mouseover);
  mainList.addEventListener("mouseout", navListHandler.mouseout);
  subList.addEventListener("mouseover", navListHandler.mouseover);
  subList.addEventListener("mouseout", navListHandler.mouseout);
}


const aside = document.querySelector("aside");

// 메뉴event 추가 및 showcontent제어 함수 실행하는 생성자함수 
function AddShowEvent(kind, content) {
  kind.addEventListener("click", () => {
      handleShowContent(content);
    });
}

// 인기 showcontent handler
const best = document.querySelector(".best");
const bestContent = document.querySelector(".bestContent");
const bestEvent = new AddShowEvent(best, bestContent);

// 상의 showcontent handler
const top_ = document.querySelector(".top");
const topContent = document.querySelector(".topContent");
const topEvent = new AddShowEvent(top_, topContent);

// 아우터 showcontent handler
const outer = document.querySelector(".outer");
const outerContent = document.querySelector(".outerContent");
const outerEvent = new AddShowEvent(outer, outerContent);

// 바지 showcontent handler
const pants = document.querySelector(".pants");
const pantsContent = document.querySelector(".pantsContent");
const pantsEvent = new AddShowEvent(pants, pantsContent);

// 기존에 content가 열려있는지 확인 및 분간하기 위한 변수
let opened = 0; 

// showcontent handler
function handleShowContent(menu) {
  if (menu.classList.contains("fixation")) {
    menu.classList.add("reverse");
  } else {
    menu.classList.add("animate");
    // 한번에 두개를 빠르게 눌러 둘다 open이 되는 오류 제어
    aside.style.pointerEvents = "none";
  }
  
  isShowAnother(menu, opened);

  menu.addEventListener("animationend", handleFixation);

  function handleFixation() {
    if (this.classList.contains("animate")) {
      this.classList.add("fixation");
      //열려있는 상태(fixation상태)시 opened변수에 할당
      //다른 show event 실행시 기존에 content가 열려있는지 확인하기 위함
      switch(this) {
        case bestContent:
          opened = bestContent;
          break;
        case topContent:
          opened = topContent;
          break;
        case outerContent:
          opened = outerContent;
          break;
        case pantsContent:
          opened = pantsContent;
          break;
        default:
          break;
      }
      this.classList.remove("animate");
      // 한번에 두개를 빠르게 눌러 둘다 open이 되는 오류 제어
      aside.style.pointerEvents = "auto";
    } else if (this.classList.contains("reverse")) {
      this.classList.remove("fixation");
      this.classList.remove("reverse");
    }
  }

  // 중복으로 열려있지 않도록 하는 함수
  // a를 누르고 b를 누르면 b는 open, 기존의 a는 close
  function isShowAnother(menu, opened) {
    if (opened === 0 || opened === menu) {
      return;
    } else if (opened.classList.contains("fixation")) {
        opened.classList.add("reverse");
        opened.addEventListener("animationend", closeOpened);
        opened.removeEventListener("animationend", closeOpened);
        // addEventListener는 추가되면 계속 적용되므로 조건부적용을 위하면
        // 삭제해야함
      }
      function closeOpened() {
        opened.classList.remove("fixation");
        opened.classList.remove("reverse");
      }
  }
}

// aside display handler
const asideHandle = document.querySelector(".asideHandler");
asideHandle.addEventListener("click", () => {
  aside.classList.toggle("none");
  asideHandle.classList.toggle("move");
}); 



