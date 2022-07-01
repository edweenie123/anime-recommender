<template>
  <div class="input-group">
    <input
      @keyup.enter="onEnter"
      v-model="text"
      placeholder="Enter MAL username"
      class="form-control"
    />
    <button @click="onEnter" type="button" class="btn btn-primary">
      <i class="fas fa-search"></i>
    </button>
  </div>
</template>
<script>
export default {
  name: "SearchBar",
  data() {
    return {
      text: "",
    };
  },
  methods: {
    async onEnter() {
      const path = `http://localhost:5000/prediction/${this.text}`
      const res = await fetch(path)
      const bruh = await res.json()
      console.log(bruh)
    },
  },
};
</script>

<style scoped>
/* input { */
/*   display: block; */
/*   width: 350px; */
/*   margin: 20px auto; */
/*   padding: 10px 45px; */
/*   /1* background: white url("assets/search-icon.svg") no-repeat 15px center; *1/ */
/*   background-size: 15px 15px; */
/*   font-size: 16px; */
/*   border: none; */
/*   border-radius: 5px; */
/*   box-shadow: rgba(50, 50, 93, 0.25) 0px 2px 5px -1px, */
/*     rgba(0, 0, 0, 0.3) 0px 1px 3px -1px; */
/* } */
</style>
