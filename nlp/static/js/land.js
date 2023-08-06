let mainText = document.querySelector('.main_txt'); //인프라
let kbText = document.querySelector('.kb'); //인프라
let sub_txt = document.querySelector(".sub_txt");
let button_index = document.querySelector(".button_index");

let target = document.querySelector("#dynamic");


let mainText1 = document.querySelector('.main_txt1'); //두번쨰 화면
let main_txt2 = document.querySelector('.main_txt2');
let photo1 = document.querySelector("#photo");

function randomString(){
    let stringArr = ["나의 상황에 맞는 카드를 추천해줘", "카페에 혜택이 많은 카드가 뭐야?", "사회초년생에게 적합한 대출상품을 알려줘","나에게 맞는 적금상품이 궁금해","비슷한 카드가 무엇인지 말해줘"];
    let selectString = stringArr[Math.floor(Math.random() * stringArr.length)];
    let selectStringArr = selectString.split(""); //한글자씩 배열로 만듬

    return selectStringArr;
}
//한글자씩 텍스트 출력 함수
function dynamic(randomArr){
    if(randomArr.length > 0){
        target.textContent += randomArr.shift();
        setTimeout(function(){
            dynamic(randomArr);
        },80);
    }else{
        setTimeout(resetTyping, 2000);
    }
}

function resetTyping(){
    target.textContent="";
    dynamic(randomString());
}

dynamic(randomString());

//커서 깜빡임 효과
function blink(){
    target.classList.toggle("active");
}
setInterval(blink,500);

mainText.style.animation='main_slide 1s ease-out forwards';
kbText.style.animation='main_slide 1s ease-out forwards';
sub_txt.style.animation='main_slide 1s ease-out forwards';
button_index.style.animation='main_slide 1s ease-out forwards';


window.addEventListener('scroll',function(){ //스크롤 되면
    let value = window.scrollY;   //스크롤될때마다 y좌표 들고오기
    console.log("scrollY",value); 

    if(value>130){
        mainText.style.animation='main_disappear 1s ease-out forwards';
        kbText.style.animation='main_disappear 1s ease-out forwards';
        // mainimg.style.animation='img_disappear 1s ease-out forwards';
        sub_txt.style.animation='main_disappear 1s ease-out forwards';
        button_index.style.animation='main_disappear 1s ease-out forwards';
        
    } /*forwards는 계속사라지게*/
    else{
        mainText.style.animation='main_slide 1s ease-out forwards';
        kbText.style.animation='main_slide 1s ease-out forwards';
        // mainimg.style.animation='img_slide 1s ease-out forwards';
        sub_txt.style.animation='main_slide 1s ease-out forwards';
        button_index.style.animation='main_slide 1s ease-out forwards';

    }

    if(value < 700 | value > 1500){
        mainText1.style.animation='text_disappear 1.5s ease-out forwards';
        main_txt2.style.animation='text_disappear 1.5s ease-out forwards';
        photo1.style.animation='img_disappear 1.5s ease-out forwards';
        
    }
    else{
        mainText1.style.animation='text_slide 1.5s ease-out forwards';
        main_txt2.style.animation='text_slide 1.5s ease-out forwards';
        photo1.style.animation='img_slide 1.5s ease-out forwards';
    }
})

