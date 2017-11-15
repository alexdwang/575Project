var favorites = [];
var recommend = [];
var movieform = []

// $(document).ready(function() {
//   movieform = $("#movieform")[0];
//   $("#recommend").click(function () {
//     $("#olmovielist").empty();
//     console.log(recommend.length);
//     for (var i = 0; i < recommend.length; i++) {
//       console.log(recommend[i]["name"] + " " + recommend[i]["genre"]);
//       $("#olmovielist").append("<li class='list-group-item'>" + recommend[i]["name"] + " " + recommend[i]["genre"] +"</li>");
//
//     }
//     movieform.reset();
//     event.preventDefault();
//     return false;
//   });
// $(document).ready(function() {
//   movieform = $("#movieform")[0];
// });

$("#search").click(function (event) {
  console.log("search");
  movieform = $("#movieform")[0];
  $("#olmovielist").empty();
  var params = {"algorithm": $("#algorithm").val(),"userid": $("#userid").val()};
  $.ajax({
      type: "POST",
      data: params,
      dataType : "json",
      success: function(respMsg){
        favorites = respMsg;
        for (var i = 0; i < favorites.length; i++) {
          var movie = favorites[i]["name"] + " " + favorites[i]["genre"];
          $("#olmovielist").append("<li class='list-group-item'>" + movie +"</li>");
        }
      }
   });
   movieform.reset();
   event.preventDefault();
   return false;
  });

$("#recommend").click(function (event) {
  $("#olmovielist").empty();
  movieform = $("#movieform")[0];
  var params = {"algorithm": $("#algorithm").val(),"movieid": $("#movieid").val()};
  $.ajax({
      type: "POST",
      data: params,
      dataType : "json",
      success: function(respMsg){
        recommend = respMsg;
        for (var i = 0; i < recommend.length; i++) {
          var movie = recommend[i]["name"] + " " + recommend[i]["genre"];
          $("#olmovielist").append("<li class='list-group-item'>" + movie +"</li>");
        }
      }
   });
   movieform.reset();
   event.preventDefault();
   return false;
  });
