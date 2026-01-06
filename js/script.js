console.log("Hello world!");

const myName = "Bharath Kumar S P";
const h1 = document.querySelector(".main-header");
console.log(myName);
console.log(h1);

// To modify the content of HTML
// h1.textContent = myName;

// To modify the CSS
// h1.style.backgroundColor = "red";

// h1.addEventListener("click", function () {
//   h1.textContent = myName;
//   h1.style.backgroundColor = "red";
//   h1.style.padding = "5rem";
// });

///////////////////////////////////////////////////////////
// Set current year
const yearEl = document.querySelector(".year");
const currentYear = new Date().getFullYear();
yearEl.textContent = currentYear;
