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

  movieform = $("#movieform")[0];
  $("#olmovielist").empty();
  userid = $("#userid").val();
  var params = {"algorithm": $('input[name=algorithm]:checked', '#movieform').val(),"userid": userid};
  $.ajax({
      url:"api/getInfo",
      type: "POST",
      data: JSON.stringify(params),
      processData: false,
      contentType: 'application/json',
      success: function(respMsg){
        favorites = respMsg[userid];
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
  movieid = $("#movieid").val();
  var params = {"algorithm": $('input[name=algorithm]:checked', '#movieform').val(), "movieid": movieid};
  $.ajax({
      url:"api/getInfo",
      type: "POST",
      data: JSON.stringify(params),
      processData: false,
      contentType: 'application/json',
      success: function(respMsg){
        recommend = respMsg[movieid];
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
