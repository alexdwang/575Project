var favorites = [];
var recommend = [];
var movieform = []

$(document).ready(function() {
  movieform = $("#movieform")[0];
  $("#recommend").click(function () {
    $("#olmovielist").empty();
    console.log(recommend.length);
    for (var i = 0; i < recommend.length; i++) {
      console.log(recommend[i]["name"] + " " + recommend[i]["genre"]);
      $("#olmovielist").append("<li class='list-group-item'>" + recommend[i]["name"] + " " + recommend[i]["genre"] +"</li>");

    }
    movieform.reset();
    event.preventDefault();
    return false;
  });

$("#search").click(function () {
  $("#olmovielist").empty();
  var params = {"algorithm": $("#algorithm").val(),"userid": $("#userid").val()};
  $.ajax({
      type: "POST",
      data: params,
      dataType : "json",
      success: function(respMsg){
        favorites = respMsg["favorites"];
        recommend = respMsg["recommend"];
        for (var i = 0; i < favorites.length; i++) {
          var movie = favorites[i]["name"] + " " + favorites[i]["genre"];
          $("#olmovielist").append("<li class='list-group-item'>" + movie +"</li>");
        }
      }
   });
   event.preventDefault();
   movieform.reset();
   return false;
  });
});
