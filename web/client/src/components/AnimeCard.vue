<template>
  <!-- <div class="hover-overlay"> -->
  <!-- </div> -->

  <!-- <div class="hover-overlay"> -->
  <!--   <!-1- <img -1-> -->
  <!--   <!-1-   src="https://mdbcdn.b-cdn.net/img/new/standard/city/053.webp" -1-> -->
  <!--   <!-1-   class="w-100" -1-> -->
  <!--   <!-1- /> -1-> -->
  <!--     <img :src="imgSrc" class="card-img-top" /> -->
  <!--     <div class="card-body"> -->
  <!--       <h5 class="card-title">{{ title }}</h5> -->
  <!--       <p class="card-text">Predicted Score: {{ predScore }}</p> -->
  <!--       <!-1- <a href="#!" class="btn btn-primary">Button</a> -1-> -->
  <!--     </div> -->
  <!--   <div class="mask"></div> -->
  <!--   </div> -->

  <!-- </div> -->
  <div class="hover-overlay">
    <div class="anime-card shadow">
      <img :src="imgSrc" />
      <div class="info">
        <h6>{{ title }}</h6>
        <p class="text-primary">Predicted Score: <span class="score">{{ predScore }}</span></p>
      </div>

      <div class="mask">
        <a :href="link" target="_blank">
          <span class="link"></span>
        </a>
      </div>
    </div>
  </div>
</template>
<script>
export default {
  props: {
    imgSrc: String,
    title: String,
    predScore: Number,
    link: String,
  },
  name: "AnimeCard",
};
</script>
<style scoped>
.hover-overlay {
  display: inline-block;
  /* border-radius: 5px; */
}

.card {
  display: inline-block;
  width: 10rem;
}

img {
  width: 100%;
}
.anime-card {
  border-radius: 5px;
  display: inline-block;
  width: 14rem;
  margin: 1rem;
  position: relative;
}

.info {
  padding: 0.5rem;
  color: white;
  height: 5rem;
  background-color: rgba(0, 0, 0, 0.8);
  position: absolute;
  bottom: 0px;
  /* z-index: 2; */
  width: 100%;
}

h6 {
  font-size: 17px;
}


p {
  font-size: 17px;
  color: rgba(244, 170, 255);
  font-weight: bold;
}

.score {

}

.link {
  position: absolute;
  width: 100%;
  height: 100%;
  top: 0;
  left: 0;
  /* z-index: 1; */
}

/* .card-body { */
/*   display: inline-block; */
/* } */

.mask {
  border-radius: 5px;
  /* background: linear-gradient( */
  /*   45deg, */
  /*   rgba(29, 236, 197, 0.5), */
  /*   rgba(91, 14, 214, 0.5) 100% */
  /* ); */
  /* pointer-events: none; */
  background-color: rgba(251, 251, 251, 0.4);
}
</style>
